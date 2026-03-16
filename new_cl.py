import os
import pandas as pd
import plotly.express as px

# Alleen dit bestand gebruiken
TARGET_FILE = os.path.join(
    ".", "data", "metingen", "20260316_v9058_Zoutmetingen_IJsselmeer.xlsx"
)

# Overzichtssheet overslaan; dit is geen meetsheet
SKIP_SHEETS = {"IJsselmeer"}

# Bestaande CSV behouden en nieuwe metingen eraan toevoegen
KEEP_EXISTING_CSV = True
CSV_PATH = os.path.join("data", "chloridemetingen ijsselmeer.csv")


def format_dutch_date(value):
    """Geef een datum terug als DD-MM-YYYY of 'onbekende datum'."""
    if pd.isna(value):
        return "onbekende datum"
    return pd.Timestamp(value).strftime("%d-%m-%Y")


def parse_mixed_datetime(values):
    """Parseer datums robuust.

    Volgorde:
    1) standaard 'mixed' parsing zonder dayfirst (werkt goed voor ISO en de bronbestanden)
    2) alleen voor resterende lege waarden: fallback met dayfirst=True
    """
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values)

    parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            series.loc[mask], format="mixed", dayfirst=True, errors="coerce"
        )
    return parsed


def extract_data_from_sheet(xlsx, sheet):
    # Extract info of measurement
    info = pd.read_excel(xlsx, sheet_name=sheet, usecols="A:B", nrows=9)
    info = info.drop([4, 5], errors="ignore")
    info.columns = ["Parameter", "Waarde"]
    info.index = info["Parameter"]
    info = info.drop("Parameter", axis=1)
    info = info.transpose()

    # Extract observations
    # Ondersteunt twee formaten:
    # 1) 6 kolommen: aparte kolommen voor Datum en Tijd (UTC)
    # 2) 5 kolommen: gecombineerde kolom 'Datum/Tijd UTC' en geen header voor diepte
    data = pd.read_excel(xlsx, sheet_name=sheet, header=10)
    data = data.drop(0, errors="ignore")
    data = data.dropna(how="all")

    if len(data.columns) == 6:
        data.columns = [
            "Diepte (m)",
            "Temperatuur (graden Celsius)",
            "Geleidendheid (mS/cm)",
            "Chloriniteit (mg/l)",
            "Datum",
            "Tijd (UTC)",
        ]
        data = data.dropna(
            subset=["Diepte (m)", "Chloriniteit (mg/l)", "Datum", "Tijd (UTC)"]
        )

        df = pd.merge(info, data, how="cross")
        df["Datum"] = parse_mixed_datetime(df["Datum"]).dt.normalize()

        tijd_as_str = df["Tijd (UTC)"].astype(str).str.strip()
        tijd_as_str = tijd_as_str.str.replace(" UTC", "", regex=False)
        df["Timedelta"] = pd.to_timedelta(tijd_as_str, errors="coerce")
        df["Datumtijd"] = df["Datum"] + df["Timedelta"]
        df = df.drop("Timedelta", axis=1)
        return df

    if len(data.columns) == 5:
        data.columns = [
            "Diepte (m)",
            "Temperatuur (graden Celsius)",
            "Geleidendheid (mS/cm)",
            "Chloriniteit (mg/l)",
            "Datum/Tijd UTC",
        ]
        data = data.dropna(
            subset=["Diepte (m)", "Chloriniteit (mg/l)", "Datum/Tijd UTC"]
        )

        # In dit bronbestand staat Datum/Tijd UTC vaak als Excel-datumgetal.
        # Converteer eerst numerieke waarden, daarna eventueel tekstuele waarden.
        data["Datumtijd"] = pd.NaT
        numeric_dt = pd.to_numeric(data["Datum/Tijd UTC"], errors="coerce")
        mask_numeric = numeric_dt.notna()
        if mask_numeric.any():
            data.loc[mask_numeric, "Datumtijd"] = pd.to_datetime(
                numeric_dt[mask_numeric],
                unit="D",
                origin="1899-12-30",
                errors="coerce",
            )

        mask_remaining = data["Datumtijd"].isna()
        if mask_remaining.any():
            data.loc[mask_remaining, "Datumtijd"] = parse_mixed_datetime(
                data.loc[mask_remaining, "Datum/Tijd UTC"]
            )

        data["Datum"] = data["Datumtijd"].dt.normalize()
        data["Tijd (UTC)"] = data["Datumtijd"].dt.strftime("%H:%M:%S")
        data = data.drop(columns=["Datum/Tijd UTC"])
        df = pd.merge(info, data, how="cross")
        return df

    raise ValueError(
        f"Sheet '{sheet}' heeft een onverwacht aantal kolommen: {len(data.columns)}. "
        f"Ingelezen kolommen: {list(data.columns)}"
    )


def merge_with_existing_csv(df_new):
    """Behoud oude waarden in de CSV en voeg nieuwe toe.

    Belangrijk: rijen uit hetzelfde bronbestand worden eerst verwijderd uit de
    bestaande CSV. Daarmee vervang je eerdere foutieve versies (zoals 03-12-2026)
    door de nieuw berekende correcte data (12-03-2026).
    """
    for col in ["Datum", "Datumtijd"]:
        if col in df_new.columns:
            df_new[col] = parse_mixed_datetime(df_new[col])

    if KEEP_EXISTING_CSV and os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH)
        for col in ["Datum", "Datumtijd"]:
            if col in df_existing.columns:
                df_existing[col] = parse_mixed_datetime(df_existing[col])

        # Verwijder oude rijen van hetzelfde bronbestand voordat nieuwe worden toegevoegd
        if "filename" in df_existing.columns and "filename" in df_new.columns:
            new_filenames = set(df_new["filename"].dropna().astype(str).unique())
            df_existing = df_existing[~df_existing["filename"].astype(str).isin(new_filenames)]

        df_combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
        df_combined = df_combined.drop_duplicates()
        return df_combined

    return df_new


def build_measurement_csv():
    if not os.path.exists(TARGET_FILE):
        raise FileNotFoundError(f"Bestand niet gevonden: {TARGET_FILE}")

    df_compleet = pd.DataFrame()
    with pd.ExcelFile(TARGET_FILE) as xlsx:
        for sheet in sorted(xlsx.sheet_names):
            if sheet in SKIP_SHEETS:
                print(f"Sheet overgeslagen: {sheet}")
                continue

            df_single = extract_data_from_sheet(xlsx, sheet)
            if df_single.empty:
                continue

            df_single["sheet"] = sheet
            df_single["filename"] = os.path.basename(TARGET_FILE)
            df_compleet = pd.concat([df_compleet, df_single], ignore_index=True)

    rename_dict = {
        "15_MG": "MG_15",
        "16_MG": "MG_16",
        "17_MG": "MG_17",
        "18_MG": "MG_18",
        "19_MG": "MG_19",
        "VG_DO_14,25": "VG_DO_14.25",
    }
    df_compleet = df_compleet.replace({"sheet": rename_dict})

    df_compleet.loc[
        (
            (df_compleet["x-coordinaat (RD)"] == "563148")
            & (df_compleet["sheet"] == "VG_KWZ_VK9")
        ),
        "x-coordinaat (RD)",
    ] = "152080"

    # Bestaande CSV behouden en nieuwe data erbij zetten
    df_output = merge_with_existing_csv(df_compleet)
    df_output.to_csv(CSV_PATH, index=False)
    print(f"CSV bijgewerkt: {CSV_PATH}")
    print(f"Aantal rijen in CSV: {len(df_output)}")


def create_visualisations():
    df = pd.read_csv(CSV_PATH)
    df["Datum"] = parse_mixed_datetime(df["Datum"])
    df["Datumtijd"] = parse_mixed_datetime(df["Datumtijd"])

    locations = df["sheet"].dropna().unique()
    folder_path = os.path.join("data", "2d visualisaties")
    os.makedirs(folder_path, exist_ok=True)

    # LET OP: bestaande HTML-bestanden worden bewust NIET verwijderd.
    for location in locations:
        df1 = df[df["sheet"] == location].copy()
        if df1.empty:
            continue

        df2 = df1[
            [
                "x-coordinaat (RD)",
                "y-coordinaat (RD)",
                "Diepte (m)",
                "Chloriniteit (mg/l)",
                "Datum",
                "Datumtijd",
            ]
        ].copy()

        df2 = df2.sort_values(by="Datumtijd")
        df2["Diepte (m)"] = df2["Diepte (m)"] * -1
        df2["Datum"] = parse_mixed_datetime(df2["Datum"])

        latest_dt = df2["Datum"].dropna().max()
        latest_date_str = format_dutch_date(latest_dt)

        df2["Datum_label"] = df2["Datum"].apply(format_dutch_date)
        ordered_dates = [
            d.strftime("%d-%m-%Y")
            for d in sorted(pd.DatetimeIndex(df2["Datum"].dropna().unique()))
        ]
        if df2["Datum_label"].eq("onbekende datum").any():
            ordered_dates.append("onbekende datum")

        fig = px.line(
            df2,
            x="Chloriniteit (mg/l)",
            y="Diepte (m)",
            title=f"Metingen op locatie {location} tot {latest_date_str}",
            color="Datum_label",
            category_orders={"Datum_label": ordered_dates},
        )
        fig.update_layout(legend_title_text="Datum")

        safe_location = (
            str(location)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "-")
        )
        output_file = os.path.join(
            folder_path,
            f"visualisatie van {safe_location} tot {latest_date_str}.html",
        )
        fig.write_html(output_file)

    print("Maken 2d visualisaties afgerond")


def main():
    build_measurement_csv()
    create_visualisations()


if __name__ == "__main__":
    main()
