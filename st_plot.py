import os
import pandas as pd
import streamlit as st
import plotly.express as px
from pyproj import Transformer

SINGLE_CSV_PATH = './data/chloridemetingen ijsselmeer.csv'

def load_single_csv():
    df = pd.read_csv(SINGLE_CSV_PATH)
    df['Datumtijd'] = pd.to_datetime(df['Datumtijd'], errors='coerce')
    df['Datum'] = df['Datumtijd'].dt.normalize()

    numeric_columns = [
        'x-coordinaat (RD)',
        'y-coordinaat (RD)',
        'Diepte (m)',
        'Temperatuur (graden Celsius)',
        'Geleidendheid (mS/cm)',
        'Chloriniteit (mg/l)',
        'Maximale diepte [m]',
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def build_summary_tables(df):
    agg_common = {
        'x-coordinaat (RD)': 'mean',
        'y-coordinaat (RD)': 'mean',
        'Chloriniteit (mg/l)': 'mean',
        'Datumtijd': 'first',
        'filename': 'first' if 'filename' in df.columns else 'first',
    }

    df_gemiddeld = df.groupby('sheet').agg(agg_common).reset_index()

    agg_max = agg_common.copy()
    agg_max['Chloriniteit (mg/l)'] = 'max'
    df_maximaal = df.groupby('sheet').agg(agg_max).reset_index()

    return df_gemiddeld, df_maximaal

def add_coordinates(df):
    transformer = Transformer.from_crs('EPSG:28992', 'EPSG:4326', always_xy=True)
    lons, lats = transformer.transform(df['x-coordinaat (RD)'].values, df['y-coordinaat (RD)'].values)
    df['lon'] = lons
    df['lat'] = lats
    return df

def add_labels_and_hover(df):
    df['label_text'] = df['Chloriniteit (mg/l)'].round(1).astype(str)
    bestandsnaam = df['filename'].fillna('') if 'filename' in df.columns else ''
    df['hover_text'] = (
        'Bestandsnaam: ' + bestandsnaam.astype(str) + '<br>' +
        'Datumtijd: ' + df['Datumtijd'].dt.strftime('%d-%m-%Y %H:%M:%S') + '<br>' +
        'Chloriniteit: ' + df['Chloriniteit (mg/l)'].round(2).astype(str) + ' mg/L'
    )
    return df

def create_map(df, title):
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
        color_continuous_scale='Viridis',
        zoom=10,
        center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()),
        map_style='carto-positron',
        title=title,
        labels={'Chloriniteit (mg/l)': 'Chloriniteit (mg/L)'},
    )

    fig.update_traces(
        marker=dict(size=12),
        textposition='top center',
        textfont=dict(size=10, color='black'),
        hovertemplate='%{customdata[0]}',
        customdata=df[['hover_text']].values,
    )
    fig.update_layout(height=800)
    return fig

st.title('Kaartvisualisatie chloride IJsselmeer')

df = load_single_csv()
datums = sorted(df['Datum'].dropna().unique())

gekozen_datum = st.selectbox(
    'Kies een meetdatum',
    datums,
    format_func=lambda d: pd.Timestamp(d).strftime('%d-%m-%Y')
)

samenvatting_type = st.radio(
    'Kies kaarttype',
    ['Gemiddelde', 'Maximale'],
    horizontal=True
)

df_dag = df[df['Datum'] == gekozen_datum].copy()
df_gemiddeld, df_maximaal = build_summary_tables(df_dag)

if samenvatting_type == 'Gemiddelde':
    df_plot = df_gemiddeld
else:
    df_plot = df_maximaal

df_plot = add_coordinates(df_plot)
df_plot = add_labels_and_hover(df_plot)

datum_str = pd.Timestamp(gekozen_datum).strftime('%d-%m-%Y')
titel = f'{samenvatting_type} Chloride waarden - {datum_str}'

fig = create_map(df_plot, titel)
st.plotly_chart(fig, use_container_width=True)