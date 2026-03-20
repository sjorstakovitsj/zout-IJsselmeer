import os
import re
import urllib.parse
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pyproj import Transformer

st.set_page_config(layout='wide')

SINGLE_CSV_PATH = './data/chloridemetingen ijsselmeer.csv'
WMS_BASE_URL = 'https://geo.rijkswaterstaat.nl/services/ogc/gdr/bodemhoogte_ijsselmeergebied/ows'
WMS_LAYER_NAME = 'bodemhoogte_ijg_2022'
WMS_ATTRIBUTION = 'Rijkswaterstaat bathymetrie IJsselmeergebied'
FIGURE_HEIGHT = 980
KAARTTYPE_OPTIES = ['Gemiddelde', 'Minimale', 'Maximale', 'Max-min']

# Duidelijker chloriniteitspalet met vaste drempels t/m 200 mg/L en
# dynamische verdeling daarboven voor betere onderscheidbaarheid.
CHLORINITY_BASE_BANDS = [
    {'lower': None, 'upper': 100.0, 'color': '#006400', 'label': '< 100 mg/L'},       # donkergroen
    {'lower': 100.0, 'upper': 150.0, 'color': '#008000', 'label': '100 - 150 mg/L'},  # groen
    {'lower': 150.0, 'upper': 200.0, 'color': '#B8860B', 'label': '150 - 200 mg/L'},  # donkergeel
]

CHLORINITY_DYNAMIC_COLORS = [
    ('#FFD700', 'vanaf 200 mg/L (geel)'),  # geel
    ('#FF8C00', 'donkeroranje'),
    ('#FFA500', 'oranje'),
    ('#8B0000', 'donkerrood'),
    ('#FF0000', 'rood'),
    ('#4B0082', 'donkerpaars'),
    ('#800080', 'paars'),
    ('#654321', 'donkerbruin'),
    ('#A52A2A', 'bruin'),
]


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
    """Normaliseer RD-coördinaten consequent naar 6 cijfers waar mogelijk.

    Regels:
    - verwijder spaties en scheidingstekens;
    - gebruik alleen de cijfers uit de invoer;
    - neem bij 6 of meer cijfers altijd de eerste 6 cijfers;
    - behoud kortere numerieke waarden om onnodig dataverlies te voorkomen.
    """
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text == '':
        return pd.NA
    digits = re.sub(r'\D', '', text)
    if digits == '':
        return pd.NA
    if len(digits) >= 6:
        return digits[:6]
    return digits


def normalize_rd_coordinate_numeric(value):
    """Normaliseer een numerieke RD-coördinaat (ook na aggregatie) naar 6 cijfers."""
    if pd.isna(value):
        return pd.NA
    try:
        integer_text = str(int(float(value)))
    except Exception:
        return pd.NA
    normalized = normalize_rd_coordinate_value(integer_text)
    return pd.to_numeric(normalized, errors='coerce')


def normalize_rd_coordinate_series(series):
    """Normaliseer een serie RD-coördinaten naar 6-cijferige numerieke waarden."""
    return pd.to_numeric(series.apply(normalize_rd_coordinate_numeric), errors='coerce')


def parse_numeric_series(series, coordinate=False):
    if coordinate:
        normalized = series.apply(normalize_rd_coordinate_value)
        return pd.to_numeric(normalized, errors='coerce')
    cleaned = series.astype(str).str.replace(' ', '', regex=False).str.strip()
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


def _format_tick_value(value: float) -> str:
    if pd.isna(value):
        return ''
    rounded = round(float(value), 1)
    if rounded.is_integer():
        return str(int(rounded))
    return f'{rounded:.1f}'


def _build_step_colorscale(intervals, vmin: float, vmax: float):
    """Maak een Plotly stepped colorscale op basis van intervallen."""
    if not intervals:
        return [(0.0, '#006400'), (1.0, '#006400')]

    if vmax <= vmin:
        color = intervals[0][2]
        return [(0.0, color), (1.0, color)]

    colorscale = []
    for start, end, color in intervals:
        if end <= start:
            continue
        start_frac = max(0.0, min(1.0, (start - vmin) / (vmax - vmin)))
        end_frac = max(0.0, min(1.0, (end - vmin) / (vmax - vmin)))
        if not colorscale:
            colorscale.append((start_frac, color))
        elif colorscale[-1][0] != start_frac or colorscale[-1][1] != color:
            colorscale.append((start_frac, color))
        colorscale.append((end_frac, color))

    if not colorscale:
        return [(0.0, '#006400'), (1.0, '#006400')]

    if colorscale[0][0] > 0.0:
        colorscale.insert(0, (0.0, colorscale[0][1]))
    if colorscale[-1][0] < 1.0:
        colorscale.append((1.0, colorscale[-1][1]))
    return colorscale


def build_chlorinity_color_config(series: pd.Series):
    """Maak een onderscheidende chloriniteitsschaal.

    Specificatie:
    - < 100 mg/L: donkergroen
    - 100 - 150 mg/L: groen
    - 150 - 200 mg/L: donkergeel
    - vanaf 200 mg/L: eerst geel, daarna dynamisch verdeeld over
      donkeroranje, oranje, donkerrood, rood, donkerpaars, paars,
      donkerbruin en bruin.
    """
    valid = pd.to_numeric(series, errors='coerce').dropna()
    if valid.empty:
        fallback = [(0.0, '#006400'), (1.0, '#A52A2A')]
        return fallback, None, None

    vmin = float(valid.min())
    vmax = float(valid.max())

    if vmax <= vmin:
        if vmin < 100.0:
            color = '#006400'
        elif vmin < 150.0:
            color = '#008000'
        elif vmin < 200.0:
            color = '#B8860B'
        else:
            color = '#FFD700'
        return [(0.0, color), (1.0, color)], [vmin, vmax], {
            'tickmode': 'array',
            'tickvals': [vmin],
            'ticktext': [_format_tick_value(vmin)],
        }

    intervals = []
    tick_values = [vmin, vmax]

    for band in CHLORINITY_BASE_BANDS:
        lower = vmin if band['lower'] is None else max(vmin, band['lower'])
        upper_limit = band['upper'] if band['upper'] is not None else vmax
        upper = min(vmax, upper_limit)
        if upper > lower:
            intervals.append((lower, upper, band['color']))
            if band['upper'] is not None and vmin < band['upper'] < vmax:
                tick_values.append(float(band['upper']))

    if vmax > 200.0:
        dynamic_start = max(vmin, 200.0)
        dynamic_count = len(CHLORINITY_DYNAMIC_COLORS)
        dynamic_edges = np.linspace(dynamic_start, vmax, dynamic_count + 1)
        for idx, (color, _label) in enumerate(CHLORINITY_DYNAMIC_COLORS):
            start = float(dynamic_edges[idx])
            end = float(dynamic_edges[idx + 1])
            if end <= start:
                continue
            intervals.append((start, end, color))
            if idx == 0 and start not in tick_values:
                tick_values.append(start)
            if idx < dynamic_count - 1:
                tick_values.append(end)
    elif vmin >= 200.0:
        # Volledig dynamische schaal wanneer alle waarden >= 200 mg/L zijn.
        dynamic_count = len(CHLORINITY_DYNAMIC_COLORS)
        dynamic_edges = np.linspace(vmin, vmax, dynamic_count + 1)
        for idx, (color, _label) in enumerate(CHLORINITY_DYNAMIC_COLORS):
            start = float(dynamic_edges[idx])
            end = float(dynamic_edges[idx + 1])
            if end <= start:
                continue
            intervals.append((start, end, color))
            if idx < dynamic_count - 1:
                tick_values.append(end)

    if not intervals:
        color = '#006400'
        intervals = [(vmin, vmax, color)]

    colorscale = _build_step_colorscale(intervals, vmin, vmax)

    tick_values = sorted({round(float(v), 6) for v in tick_values if vmin <= float(v) <= vmax})
    colorbar_ticks = {
        'tickmode': 'array',
        'tickvals': tick_values,
        'ticktext': [_format_tick_value(v) for v in tick_values],
    }

    return colorscale, [vmin, vmax], colorbar_ticks


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
    ruwe data van die dag en locatie. Extra kaarttype 'Max-min' toont het
    verschil tussen de maximale en minimale chloriniteit per meetlocatie.
    """
    agg_meta = {
        'x-coordinaat (RD)': 'mean',
        'y-coordinaat (RD)': 'mean',
        'Diepte (m)': 'max',
        'Datumtijd': 'first',
    }
    if 'filename' in df.columns:
        agg_meta['filename'] = 'first'

    df_meta = df.groupby('sheet', dropna=True).agg(agg_meta).reset_index()
    for coord_col in ['x-coordinaat (RD)', 'y-coordinaat (RD)']:
        if coord_col in df_meta.columns:
            df_meta[coord_col] = normalize_rd_coordinate_series(df_meta[coord_col])
    df_stats = (
        df.groupby('sheet', dropna=True)['Chloriniteit (mg/l)']
        .agg(['mean', 'min', 'max'])
        .reset_index()
    )

    df_gemiddeld = df_meta.merge(
        df_stats[['sheet', 'mean']].rename(columns={'mean': 'Chloriniteit (mg/l)'}),
        on='sheet',
        how='left',
    )
    df_gemiddeld['waarde_label'] = 'Chloriniteit'

    df_minimaal = df_meta.merge(
        df_stats[['sheet', 'min']].rename(columns={'min': 'Chloriniteit (mg/l)'}),
        on='sheet',
        how='left',
    )
    df_minimaal['waarde_label'] = 'Chloriniteit'

    df_maximaal = df_meta.merge(
        df_stats[['sheet', 'max']].rename(columns={'max': 'Chloriniteit (mg/l)'}),
        on='sheet',
        how='left',
    )
    df_maximaal['waarde_label'] = 'Chloriniteit'

    df_max_min = df_meta.merge(
        df_stats.assign(**{'Chloriniteit (mg/l)': df_stats['max'] - df_stats['min']})[['sheet', 'Chloriniteit (mg/l)']],
        on='sheet',
        how='left',
    )
    df_max_min['waarde_label'] = 'Max-min chloriniteit'

    return df_gemiddeld, df_minimaal, df_maximaal, df_max_min


def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for coord_col in ['x-coordinaat (RD)', 'y-coordinaat (RD)']:
        if coord_col in df.columns:
            df[coord_col] = normalize_rd_coordinate_series(df[coord_col])
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
    waarde_label = (
        df['waarde_label'].fillna('Chloriniteit')
        if 'waarde_label' in df.columns
        else pd.Series(['Chloriniteit'] * len(df), index=df.index)
    )

    if 'Diepte (m)' in df.columns:
        diepte_numeric = pd.to_numeric(df['Diepte (m)'], errors='coerce')
        max_diepte_tekst = diepte_numeric.round(2).astype(str)
        max_diepte_tekst = max_diepte_tekst.where(diepte_numeric.notna(), '')
    else:
        max_diepte_tekst = pd.Series([''] * len(df), index=df.index)

    waarde_tekst = df['Chloriniteit (mg/l)'].round(2).astype(str)
    df['hover_text'] = (
        'Bestandsnaam: ' + bestandsnaam.astype(str)
        + '<br>Datumtijd: ' + datumtekst
        + '<br>' + waarde_label.astype(str) + ': ' + waarde_tekst + ' mg/L'
        + '<br>Maximaal gemeten diepte (m): ' + max_diepte_tekst.astype(str)
    )
    return df


def create_map(df: pd.DataFrame, title: str, colorbar_side: str = 'left'):
    color_scale, range_color, colorbar_ticks = build_chlorinity_color_config(df['Chloriniteit (mg/l)'])
    waarde_label = 'Chloriniteit'
    if 'waarde_label' in df.columns and not df['waarde_label'].dropna().empty:
        waarde_label = str(df['waarde_label'].dropna().iloc[0])
    colorbar_title = f'{waarde_label} (mg/L)'

    colorbar_config = {
        'title': colorbar_title,
        'y': 0.5,
        'yanchor': 'middle',
        'len': 0.88,
        'thickness': 22,
        'outlinewidth': 1,
    }
    if colorbar_ticks:
        colorbar_config.update(colorbar_ticks)

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
            'waarde_label': False,
        },
        color_continuous_scale=color_scale,
        range_color=range_color,
        zoom=8,
        center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()),
        map_style='white-bg',
        title=title,
        labels={'Chloriniteit (mg/l)': colorbar_title},
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

    df_gemiddeld, df_minimaal, df_maximaal, df_max_min = build_summary_tables(df_day)
    if samenvatting_type == 'Gemiddelde':
        df_plot = df_gemiddeld
    elif samenvatting_type == 'Minimale':
        df_plot = df_minimaal
    elif samenvatting_type == 'Maximale':
        df_plot = df_maximaal
    else:
        df_plot = df_max_min

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
        KAARTTYPE_OPTIES,
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
        KAARTTYPE_OPTIES,
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
