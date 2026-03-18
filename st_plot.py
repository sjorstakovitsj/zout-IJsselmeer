import os
import re
import urllib.parse

import pandas as pd
import plotly.express as px
import streamlit as st
from pyproj import Transformer

st.set_page_config(layout='wide')

SINGLE_CSV_PATH = './data/chloridemetingen ijsselmeer.csv'

WMS_BASE_URL = 'https://geo.rijkswaterstaat.nl/services/ogc/gdr/bodemhoogte_ijsselmeergebied/ows'
WMS_LAYER_NAME = 'bodemhoogte_ijg_2022'
WMS_ATTRIBUTION = 'Rijkswaterstaat bathymetrie IJsselmeergebied'
GREEN_THRESHOLD_MG_L = 150.0
FIGURE_HEIGHT = 980


def parse_mixed_datetime(values):
    """Parseer datums robuust voor gemengde formaten."""
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values)

    parsed = pd.to_datetime(series, format='mixed', errors='coerce')
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            series.loc[mask], format='mixed', dayfirst=True, errors='coerce'
        )
    return parsed


def normalize_rd_coordinate_value(value):
    """Normaliseer RD-coördinaten.

    Regels:
    - als de waarde alleen uit cijfers bestaat en langer is dan 6 cijfers,
      neem de eerste 6 cijfers;
    - behoud waarden met scheidingstekens (punt/komma) voor normale numerieke parsing.
    """
    if pd.isna(value):
        return value

    text = str(value).strip()
    if text == '':
        return pd.NA

    if re.fullmatch(r'\d{7,}', text):
        text = text[:6]

    return text


def parse_numeric_series(series, coordinate=False):
    if coordinate:
        series = series.apply(normalize_rd_coordinate_value)

    cleaned = series.astype(str).str.replace(' ', '', regex=False).str.strip()

    comma_no_dot_mask = cleaned.str.contains(',', na=False) & ~cleaned.str.contains('.', na=False)
    cleaned.loc[comma_no_dot_mask] = cleaned.loc[comma_no_dot_mask].str.replace(',', '.', regex=False)

    return pd.to_numeric(cleaned, errors='coerce')


def build_wms_map_source():
    return (
        f'{WMS_BASE_URL}?'
        'SERVICE=WMS'
        '&VERSION=1.3.0'
        '&REQUEST=GetMap'
        f'&LAYERS={WMS_LAYER_NAME}'
        '&STYLES='
        '&CRS=EPSG:3857'
        '&BBOX={bbox-epsg-3857}'
        '&WIDTH=1024'
        '&HEIGHT=1024'
        '&FORMAT=image/png'
        '&TRANSPARENT=true'
    )


def build_wms_legend_url():
    params = {
        'SERVICE': 'WMS',
        'REQUEST': 'GetLegendGraphic',
        'VERSION': '1.0.0',
        'FORMAT': 'image/png',
        'LAYER': WMS_LAYER_NAME,
        'STYLE': '',
    }
    return f"{WMS_BASE_URL}?{urllib.parse.urlencode(params)}"


def build_chlorinity_color_config(series: pd.Series):
    """Maak een kleurschaal waarbij:
    - waarden < 150 mg/l groen zijn;
    - waarden vanaf 150 mg/l meteen geel starten;
    - waarden boven 150 mg/l dynamisch autoschalen naar rood.
    """
    valid = pd.to_numeric(series, errors='coerce').dropna()
    if valid.empty:
        return [(0.0, 'green'), (1.0, 'red')], None

    vmin = float(valid.min())
    vmax = float(valid.max())

    if vmax <= vmin:
        if vmax < GREEN_THRESHOLD_MG_L:
            return [(0.0, 'green'), (1.0, 'green')], [vmin, vmax]
        return [(0.0, 'yellow'), (0.66, 'orange'), (1.0, 'red')], [vmin, vmax]

    if vmax < GREEN_THRESHOLD_MG_L:
        return [(0.0, 'green'), (1.0, 'green')], [vmin, vmax]

    if vmin >= GREEN_THRESHOLD_MG_L:
        return [
            (0.0, 'yellow'),
            (0.66, 'orange'),
            (1.0, 'red'),
        ], [vmin, vmax]

    threshold_fraction = (GREEN_THRESHOLD_MG_L - vmin) / (vmax - vmin)
    threshold_fraction = min(max(threshold_fraction, 0.0), 1.0)
    epsilon = min(1e-6, max(threshold_fraction / 1000.0, 1e-9))
    green_stop = max(0.0, threshold_fraction - epsilon)

    color_scale = [
        (0.0, 'green'),
        (green_stop, 'green'),
        (threshold_fraction, 'yellow'),
        (min(threshold_fraction + (1.0 - threshold_fraction) * 0.5, 1.0), 'orange'),
        (1.0, 'red'),
    ]
    return color_scale, [vmin, vmax]


def load_single_csv() -> pd.DataFrame:
    df = pd.read_csv(SINGLE_CSV_PATH, dtype=str, low_memory=False)

    if 'Datumtijd' in df.columns:
        df['Datumtijd'] = parse_mixed_datetime(df['Datumtijd'])
    else:
        df['Datumtijd'] = pd.NaT

    if 'Datum' in df.columns:
        parsed_datum = parse_mixed_datetime(df['Datum']).dt.normalize()
        df['Datum'] = parsed_datum
        mask_missing_dt = df['Datumtijd'].isna() & parsed_datum.notna()
        df.loc[mask_missing_dt, 'Datumtijd'] = parsed_datum.loc[mask_missing_dt]
    else:
        df['Datum'] = df['Datumtijd'].dt.normalize()

    df['Datum'] = df['Datumtijd'].dt.normalize().fillna(df['Datum'])
    df['Jaar'] = df['Datum'].dt.year

    numeric_columns = [
        'Diepte (m)',
        'Temperatuur (graden Celsius)',
        'Geleidendheid (mS/cm)',
        'Chloriniteit (mg/l)',
        'Maximale diepte [m]',
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = parse_numeric_series(df[col], coordinate=False)

    for col in ['x-coordinaat (RD)', 'y-coordinaat (RD)']:
        if col in df.columns:
            df[col] = parse_numeric_series(df[col], coordinate=True)

    return df


def build_summary_tables(df: pd.DataFrame):
    """Bouw samenvattingstabellen per meetlocatie voor één meetdag.

    Naast chloriniteit wordt ook de maximaal gemeten diepte meegenomen uit de
    ruwe data van die dag en locatie.
    """
    agg_common = {
        'x-coordinaat (RD)': 'mean',
        'y-coordinaat (RD)': 'mean',
        'Chloriniteit (mg/l)': 'mean',
        'Diepte (m)': 'max',
        'Datumtijd': 'first',
    }
    if 'filename' in df.columns:
        agg_common['filename'] = 'first'

    df_gemiddeld = df.groupby('sheet', dropna=True).agg(agg_common).reset_index()

    agg_max = agg_common.copy()
    agg_max['Chloriniteit (mg/l)'] = 'max'
    df_maximaal = df.groupby('sheet', dropna=True).agg(agg_max).reset_index()

    agg_min = agg_common.copy()
    agg_min['Chloriniteit (mg/l)'] = 'min'
    df_minimaal = df.groupby('sheet', dropna=True).agg(agg_min).reset_index()

    return df_gemiddeld, df_minimaal, df_maximaal


def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=['x-coordinaat (RD)', 'y-coordinaat (RD)'])
    if df.empty:
        return df.assign(lon=pd.Series(dtype=float), lat=pd.Series(dtype=float))

    transformer = Transformer.from_crs('EPSG:28992', 'EPSG:4326', always_xy=True)
    lons, lats = transformer.transform(
        df['x-coordinaat (RD)'].values,
        df['y-coordinaat (RD)'].values,
    )
    df['lon'] = lons
    df['lat'] = lats
    df = df.dropna(subset=['lon', 'lat'])
    return df


def add_labels_and_hover(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['label_text'] = df['Chloriniteit (mg/l)'].round(1).astype(str)
    bestandsnaam = (
        df['filename'].fillna('')
        if 'filename' in df.columns
        else pd.Series([''] * len(df), index=df.index)
    )
    datumtekst = pd.to_datetime(df['Datumtijd'], errors='coerce').dt.strftime('%d-%m-%Y %H:%M:%S').fillna('')

    if 'Diepte (m)' in df.columns:
        diepte_numeric = pd.to_numeric(df['Diepte (m)'], errors='coerce')
        max_diepte_tekst = diepte_numeric.round(2).astype(str)
        max_diepte_tekst = max_diepte_tekst.where(diepte_numeric.notna(), '')
    else:
        max_diepte_tekst = pd.Series([''] * len(df), index=df.index)

    df['hover_text'] = (
        'Bestandsnaam: ' + bestandsnaam.astype(str)
        + '<br>Datumtijd: ' + datumtekst
        + '<br>Chloriniteit: ' + df['Chloriniteit (mg/l)'].round(2).astype(str) + ' mg/L'
        + '<br>Maximaal gemeten diepte (m): ' + max_diepte_tekst.astype(str)
    )
    return df


def create_map(df: pd.DataFrame, title: str, colorbar_side: str = 'left'):
    color_scale, range_color = build_chlorinity_color_config(df['Chloriniteit (mg/l)'])

    colorbar_config = {
        'title': 'Chloriniteit (mg/L)',
        'y': 0.5,
        'yanchor': 'middle',
        'len': 0.88,
        'thickness': 22,
        'outlinewidth': 1,
    }
    if colorbar_side == 'right':
        colorbar_config.update({'x': 1.06, 'xanchor': 'left'})
        margin = {'l': 20, 'r': 90, 't': 60, 'b': 10}
    else:
        colorbar_config.update({'x': -0.16, 'xanchor': 'left'})
        margin = {'l': 100, 'r': 20, 't': 60, 'b': 10}

    fig = px.scatter_map(
        df,
        lat='lat',
        lon='lon',
        color='Chloriniteit (mg/l)',
        text='label_text',
        hover_data={
            'lon': False,
            'lat': False,
            'Chloriniteit (mg/l)': False,
            'hover_text': True,
            'label_text': False,
        },
        color_continuous_scale=color_scale,
        range_color=range_color,
        zoom=8,
        center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()),
        map_style='white-bg',
        title=title,
        labels={'Chloriniteit (mg/l)': 'Chloriniteit (mg/L)'},
    )

    fig.update_layout(
        map_layers=[
            {
                'below': 'traces',
                'sourcetype': 'raster',
                'sourceattribution': WMS_ATTRIBUTION,
                'source': [build_wms_map_source()],
            }
        ],
        height=FIGURE_HEIGHT,
        margin=margin,
        coloraxis_colorbar=colorbar_config,
    )

    fig.update_traces(
        marker=dict(size=12),
        textposition='top center',
        textfont=dict(size=10, color='black'),
        hovertemplate='%{customdata[0]}',
        customdata=df[['hover_text']].values,
    )
    return fig


def build_plot_dataframe(df_day: pd.DataFrame, samenvatting_type: str) -> pd.DataFrame:
    """Bouw de plotdata voor één geselecteerde meetdag en kaarttype."""
    if df_day.empty:
        return pd.DataFrame()

    df_gemiddeld, df_minimaal, df_maximaal = build_summary_tables(df_day)
    if samenvatting_type == 'Gemiddelde':
        df_plot = df_gemiddeld
    elif samenvatting_type == 'Minimale':
        df_plot = df_minimaal
    else:
        df_plot = df_maximaal

    if df_plot.empty:
        return pd.DataFrame()

    df_plot = add_coordinates(df_plot)
    if df_plot.empty:
        return pd.DataFrame()

    return add_labels_and_hover(df_plot)


def format_date_option(value):
    return pd.Timestamp(value).strftime('%d-%m-%Y')


st.title('Kaartvisualisatie chloride IJsselmeer')

df = load_single_csv()

jaren = sorted(df['Jaar'].dropna().astype(int).unique())
if not jaren:
    st.warning('Geen geldige meetjaren gevonden in het bestand.')
    st.stop()

gekozen_jaar = st.selectbox(
    'Kies een meetjaar',
    jaren,
    index=len(jaren) - 1,
)

df_jaar = df[df['Jaar'] == gekozen_jaar].copy()
datums = sorted(df_jaar['Datum'].dropna().unique())

if not datums:
    st.warning('Geen geldige meetdatums gevonden voor het geselecteerde jaar.')
    st.stop()

left_default_idx = max(len(datums) - 2, 0)
left_options = datums[:-1] if len(datums) > 1 else datums
if not left_options:
    left_options = datums

left_state_key = 'vergelijking_linker_datum'
if left_state_key not in st.session_state or st.session_state[left_state_key] not in left_options:
    st.session_state[left_state_key] = left_options[min(left_default_idx, len(left_options) - 1)]

left_col, right_col = st.columns(2, gap='medium')

with left_col:
    st.subheader('Linker kaart (oudere)')
    gekozen_datum_links = st.selectbox(
        'Kies oudere meetdatum',
        left_options,
        index=left_options.index(st.session_state[left_state_key]),
        format_func=format_date_option,
        key=left_state_key,
    )
    samenvatting_type_links = st.radio(
        'Kies kaarttype (links)',
        ['Gemiddelde', 'Minimale', 'Maximale'],
        index=0,
        horizontal=True,
        key='kaarttype_links',
    )

right_options = [d for d in datums if d > gekozen_datum_links]
if not right_options:
    right_options = [gekozen_datum_links]

right_state_key = 'vergelijking_rechter_datum'
right_default = datums[-1]
if right_state_key not in st.session_state or st.session_state[right_state_key] not in right_options:
    if right_default in right_options:
        st.session_state[right_state_key] = right_default
    else:
        st.session_state[right_state_key] = right_options[-1]

with right_col:
    st.subheader('Rechter kaart (recentere)')
    gekozen_datum_rechts = st.selectbox(
        'Kies recentere meetdatum',
        right_options,
        index=right_options.index(st.session_state[right_state_key]),
        format_func=format_date_option,
        key=right_state_key,
    )
    samenvatting_type_rechts = st.radio(
        'Kies kaarttype (rechts)',
        ['Gemiddelde', 'Minimale', 'Maximale'],
        index=0,
        horizontal=True,
        key='kaarttype_rechts',
    )

df_dag_links = df_jaar[df_jaar['Datum'] == gekozen_datum_links].copy()
df_plot_links = build_plot_dataframe(df_dag_links, samenvatting_type_links)
if df_plot_links.empty:
    with left_col:
        st.warning('Geen samenvattingsdata of geldige kaartcoördinaten beschikbaar voor de linker selectie.')
else:
    titel_links = f'{samenvatting_type_links} Chloride waarden - {format_date_option(gekozen_datum_links)}'
    fig_links = create_map(df_plot_links, titel_links, colorbar_side='left')
    with left_col:
        st.plotly_chart(fig_links, width='stretch')

df_dag_rechts = df_jaar[df_jaar['Datum'] == gekozen_datum_rechts].copy()
df_plot_rechts = build_plot_dataframe(df_dag_rechts, samenvatting_type_rechts)
if df_plot_rechts.empty:
    with right_col:
        st.warning('Geen samenvattingsdata of geldige kaartcoördinaten beschikbaar voor de rechter selectie.')
else:
    titel_rechts = f'{samenvatting_type_rechts} Chloride waarden - {format_date_option(gekozen_datum_rechts)}'
    fig_rechts = create_map(df_plot_rechts, titel_rechts, colorbar_side='right')
    with right_col:
        st.plotly_chart(fig_rechts, width='stretch')

legend_url = build_wms_legend_url()
with st.expander('Legenda bathymetrie', expanded=False):
    st.markdown(
        f'<img src="{legend_url}" alt="Legenda bathymetrie" style="max-width:360px; width:100%; height:auto;"/>',
        unsafe_allow_html=True,
    )
    st.caption('Legenda uit de WMS-kaartservice van Rijkswaterstaat.')
