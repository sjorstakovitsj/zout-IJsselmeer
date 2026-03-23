
import glob
import os
import re
import pandas as pd
import plotly.express as px

# Excel-bron (blijft ondersteund)
TARGET_FILE = os.path.join(
    ".", "data", "metingen", "20260323_X1272_Zoutmetingen_IJsselmeer.xlsx"
)

# Nieuwe bronmap voor losse CSV-bestanden per meetpunt
CSV_INPUT_DIR = os.path.join(".", "data", "metingen", "csv")

# Overzichtssheet overslaan; dit is geen meetsheet
SKIP_SHEETS = {"IJsselmeer"}

# Bestaande CSV behouden en nieuwe metingen eraan toevoegen
KEEP_EXISTING_CSV = True
CSV_PATH = os.path.join("data", "chloridemetingen ijsselmeer.csv")

# Maximale RD-afstand (meters) voor fallback op coördinaten als bestandsnaam niet matcht
LOCATION_COORD_TOLERANCE = 500


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


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_column(columns, candidates):
    normalized = {normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def parse_numeric_series(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip(),
        errors="coerce",
    )


def safe_sheet_name_from_filename(file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    parts = [p for p in re.split(r"[_\-\s]+", stem) if p]

    if parts and re.fullmatch(r"\d{6,8}", parts[0]):
        parts = parts[1:]

    skip_tokens = {
        "zout",
        "zoutmetingen",
        "zoutmeting",
        "chloride",
        "chloriniteit",
        "salinity",
        "salt",
        "metingen",
        "meting",
        "cl",
    }
    filtered = [p for p in parts if p.lower() not in skip_tokens]
    if filtered:
        return "_".join(filtered)
    return stem


def extract_date_token_from_name(file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    match = re.search(r"(?<!\d)(\d{6,8})(?!\d)", stem)
    return match.group(1) if match else None


def load_location_mapping(csv_dir):
    """Lees alle locatiemapping-bestanden onder de CSV-map in.

    Verwacht Excelbestanden met minimaal kolommen voor:
    - filebasename
    - Locatie
    - rdx
    - rdy

    Ondersteunt meerdere meetdagen (bijv. 260303, 260304, 260305, 260306)
    zolang er per dag een mappingbestand aanwezig is in of onder CSV_INPUT_DIR.
    """
    pattern_xlsx = os.path.join(csv_dir, "**", "*.xlsx")
    pattern_xls = os.path.join(csv_dir, "**", "*.xls")
    mapping_files = sorted(set(glob.glob(pattern_xlsx, recursive=True) + glob.glob(pattern_xls, recursive=True)))

    frames = []
    for mapping_file in mapping_files:
        try:
            xls = pd.ExcelFile(mapping_file)
        except Exception:
            continue

        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(mapping_file, sheet_name=sheet)
            except Exception:
                continue
            if df.empty:
                continue

            df.columns = [str(c).strip() for c in df.columns]
            basename_col = find_column(df.columns, ["filebasename", "bestand", "bestandnaam", "filename", "filebase"])
            location_col = find_column(df.columns, ["Locatie", "location", "meetpunt", "naam"])
            x_col = find_column(df.columns, ["rdx", "x_rd", "xcoordinaatrd", "x"])
            y_col = find_column(df.columns, ["rdy", "y_rd", "ycoordinaatrd", "y"])

            if basename_col is None or location_col is None:
                continue

            mapping = pd.DataFrame()
            mapping["filebasename"] = df[basename_col].astype(str).str.strip().str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
            mapping["Locatie"] = df[location_col].astype(str).str.strip()
            mapping["rdx"] = parse_numeric_series(df[x_col]) if x_col else pd.NA
            mapping["rdy"] = parse_numeric_series(df[y_col]) if y_col else pd.NA
            mapping["mapping_file"] = mapping_file
            mapping["mapping_date_token"] = extract_date_token_from_name(mapping_file)
            mapping = mapping.dropna(subset=["filebasename", "Locatie"]) 
            mapping = mapping[mapping["filebasename"] != ""]
            if not mapping.empty:
                frames.append(mapping)

    if not frames:
        return pd.DataFrame(columns=["filebasename", "Locatie", "rdx", "rdy", "mapping_file", "mapping_date_token"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["filebasename"], keep="last")
    return combined


def find_location_for_csv(csv_path, x_value, y_value, location_mapping):
    """Leid de locatienaam af uit mappingbestand o.b.v. bestandsnaam, met RD-fallback."""
    if location_mapping is None or location_mapping.empty:
        return None

    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    date_token = extract_date_token_from_name(csv_path)

    # 1) Exacte match op bestandsbasisnaam
    exact = location_mapping[location_mapping["filebasename"].astype(str) == csv_basename]
    if not exact.empty:
        return str(exact.iloc[0]["Locatie"]).strip()

    # 2) Zelfde meetdag + dichtstbijzijnde RD-coördinaat
    candidates = location_mapping.copy()
    if date_token and "mapping_date_token" in candidates.columns:
        same_day = candidates[candidates["mapping_date_token"].astype(str) == str(date_token)]
        if not same_day.empty:
            candidates = same_day

    candidates = candidates.dropna(subset=["rdx", "rdy", "Locatie"])
    if candidates.empty or pd.isna(x_value) or pd.isna(y_value):
        return None

    candidates = candidates.copy()
    candidates["coord_distance"] = ((candidates["rdx"] - x_value) ** 2 + (candidates["rdy"] - y_value) ** 2) ** 0.5
    nearest = candidates.sort_values("coord_distance").iloc[0]
    if pd.notna(nearest["coord_distance"]) and float(nearest["coord_distance"]) <= LOCATION_COORD_TOLERANCE:
        return str(nearest["Locatie"]).strip()

    return None


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


def read_measurement_csv(csv_path, location_mapping=None):
    """Lees een losse meetpunt-CSV en zet deze om naar hetzelfde formaat als de Excel-output."""
    try:
        data = pd.read_csv(csv_path, dtype=str)
        if len(data.columns) == 1:
            data = pd.read_csv(csv_path, sep=";", dtype=str)
    except UnicodeDecodeError:
        data = pd.read_csv(csv_path, dtype=str, encoding="latin1")
        if len(data.columns) == 1:
            data = pd.read_csv(csv_path, sep=";", dtype=str, encoding="latin1")

    data.columns = [str(c).strip() for c in data.columns]
    data = data.dropna(how="all")
    if data.empty:
        return pd.DataFrame()

    datetime_col = find_column(data.columns, ["datetime", "datumtijd", "timestamp", "date_time"])
    x_col = find_column(data.columns, ["rdx", "x_rd", "xcoordinaatrd", "x", "xcoordinaat"])
    y_col = find_column(data.columns, ["rdy", "y_rd", "ycoordinaatrd", "y", "ycoordinaat"])
    depth_col = find_column(data.columns, ["sensor_depth", "depth", "diepte", "sensordepth"])
    cond_col = find_column(data.columns, ["conductivity", "geleidendheid", "ec"])
    temp_col = find_column(data.columns, ["temperature", "temperatuur"])
    cl_col = find_column(data.columns, ["cl_rws_stdrd", "chloriniteit", "chloride", "cl"])

    required = {
        "datetime": datetime_col,
        "x": x_col,
        "y": y_col,
        "depth": depth_col,
        "conductivity": cond_col,
        "temperature": temp_col,
        "chloriniteit": cl_col,
    }
    missing = [label for label, col in required.items() if col is None]
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' mist verplichte kolommen: {missing}. Ingelezen kolommen: {list(data.columns)}"
        )

    # Verwijder eventuele rij met eenheden direct onder de header
    dt_as_text = data[datetime_col].astype(str).str.strip().str.lower()
    unit_mask = dt_as_text.str.contains("yyyy", na=False)
    unit_mask |= data[cl_col].astype(str).str.strip().str.lower().eq("mg/l")
    unit_mask |= data[x_col].astype(str).str.strip().str.lower().eq("m")
    if unit_mask.any():
        data = data.loc[~unit_mask].copy()

    data["Datumtijd"] = parse_mixed_datetime(data[datetime_col])
    data["Datum"] = data["Datumtijd"].dt.normalize()
    data["Tijd (UTC)"] = data["Datumtijd"].dt.strftime("%H:%M:%S")

    data["x-coordinaat (RD)"] = parse_numeric_series(data[x_col])
    data["y-coordinaat (RD)"] = parse_numeric_series(data[y_col])

    depth_series = parse_numeric_series(data[depth_col])
    data["Diepte (m)"] = depth_series.abs()

    conductivity_series = parse_numeric_series(data[cond_col])
    # Losse CSV's gebruiken meestal mS/m; normaliseer naar mS/cm zoals in Excel.
    if conductivity_series.dropna().median() > 100:
        conductivity_series = conductivity_series / 100.0
    data["Geleidendheid (mS/cm)"] = conductivity_series

    data["Temperatuur (graden Celsius)"] = parse_numeric_series(data[temp_col])
    data["Chloriniteit (mg/l)"] = parse_numeric_series(data[cl_col])

    data = data.dropna(
        subset=[
            "Diepte (m)",
            "Chloriniteit (mg/l)",
            "Datumtijd",
            "x-coordinaat (RD)",
            "y-coordinaat (RD)",
        ]
    )

    if data.empty:
        return pd.DataFrame()

    representative_x = data["x-coordinaat (RD)"].dropna().median()
    representative_y = data["y-coordinaat (RD)"].dropna().median()
    mapped_location = find_location_for_csv(csv_path, representative_x, representative_y, location_mapping)
    sheet_name = mapped_location or safe_sheet_name_from_filename(csv_path)

    output = data[
        [
            "x-coordinaat (RD)",
            "y-coordinaat (RD)",
            "Diepte (m)",
            "Temperatuur (graden Celsius)",
            "Geleidendheid (mS/cm)",
            "Chloriniteit (mg/l)",
            "Datum",
            "Tijd (UTC)",
            "Datumtijd",
        ]
    ].copy()
    output["Locatie"] = pd.NA
    output["Rondnr"] = sheet_name
    output["sheet"] = sheet_name
    output["filename"] = os.path.basename(csv_path)
    return output


def merge_with_existing_csv(df_new):
    """Behoud oude waarden in de CSV en voeg nieuwe toe.

    Belangrijk: rijen uit hetzelfde bronbestand worden eerst verwijderd uit de
    bestaande CSV. Daarmee vervang je eerdere foutieve versies door de nieuw
    berekende correcte data.
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
            df_existing = df_existing[
                ~df_existing["filename"].astype(str).isin(new_filenames)
            ]

        df_combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
        df_combined = df_combined.drop_duplicates()
        return df_combined

    return df_new


def collect_from_excel(target_file):
    df_compleet = pd.DataFrame()
    with pd.ExcelFile(target_file) as xlsx:
        for sheet in sorted(xlsx.sheet_names):
            if sheet in SKIP_SHEETS:
                print(f"Sheet overgeslagen: {sheet}")
                continue
            df_single = extract_data_from_sheet(xlsx, sheet)
            if df_single.empty:
                continue
            df_single["sheet"] = sheet
            df_single["filename"] = os.path.basename(target_file)
            df_compleet = pd.concat([df_compleet, df_single], ignore_index=True)
    return df_compleet


def collect_from_csv_directory(csv_dir, location_mapping=None):
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for csv_file in csv_files:
        try:
            df_single = read_measurement_csv(csv_file, location_mapping=location_mapping)
        except Exception as exc:
            print(f"CSV overgeslagen wegens fout ({csv_file}): {exc}")
            continue
        if df_single.empty:
            continue
        frames.append(df_single)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_measurement_csv():
    sources_found = False
    frames = []
    location_mapping = load_location_mapping(CSV_INPUT_DIR) if os.path.isdir(CSV_INPUT_DIR) else pd.DataFrame()

    if os.path.exists(TARGET_FILE):
        sources_found = True
        frames.append(collect_from_excel(TARGET_FILE))

    if os.path.isdir(CSV_INPUT_DIR):
        csv_files = glob.glob(os.path.join(CSV_INPUT_DIR, "**", "*.csv"), recursive=True)
        if csv_files:
            sources_found = True
            frames.append(collect_from_csv_directory(CSV_INPUT_DIR, location_mapping=location_mapping))

    if not sources_found:
        raise FileNotFoundError(
            f"Geen invoerbestanden gevonden. Verwacht Excel: {TARGET_FILE} of CSV-map: {CSV_INPUT_DIR}"
        )

    if not frames:
        raise ValueError("Er is geen meetdata ingelezen uit de gevonden bronbestanden.")

    df_compleet = pd.concat(frames, ignore_index=True, sort=False)

    rename_dict = {
        "15_MG": "MG_15",
        "16_MG": "MG_16",
        "17_MG": "MG_17",
        "18_MG": "MG_18",
        "19_MG": "MG_19",
        "VG_DO_14,25": "VG_DO_14.25",
    }
    if "sheet" in df_compleet.columns:
        df_compleet = df_compleet.replace({"sheet": rename_dict})
    if "Rondnr" in df_compleet.columns:
        df_compleet = df_compleet.replace({"Rondnr": rename_dict})

    # Historische correctie uit bestaand script behouden
    if {"x-coordinaat (RD)", "sheet"}.issubset(df_compleet.columns):
        mask_fix = (
            (df_compleet["x-coordinaat (RD)"].astype(str) == "563148")
            & (df_compleet["sheet"].astype(str) == "VG_KWZ_VK9")
        )
        if mask_fix.any():
            df_compleet.loc[mask_fix, "x-coordinaat (RD)"] = "152080"

    # Bestaande CSV behouden en nieuwe data erbij zetten
    df_output = merge_with_existing_csv(df_compleet)
    ensure_parent_dir(CSV_PATH)
    df_output.to_csv(CSV_PATH, index=False)
    print(f"CSV bijgewerkt: {CSV_PATH}")
    print(f"Aantal rijen in CSV: {len(df_output)}")


def create_visualisations():
    if not os.path.exists(CSV_PATH):
        print(f"Geen CSV gevonden voor visualisaties: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    if df.empty or "sheet" not in df.columns:
        print("Geen data beschikbaar voor visualisaties.")
        return

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

        required_cols = [
            "x-coordinaat (RD)",
            "y-coordinaat (RD)",
            "Diepte (m)",
            "Chloriniteit (mg/l)",
            "Datum",
            "Datumtijd",
        ]
        missing = [col for col in required_cols if col not in df1.columns]
        if missing:
            print(f"Visualisatie overgeslagen voor {location}; ontbrekende kolommen: {missing}")
            continue

        df2 = df1[required_cols].copy()
        df2 = df2.sort_values(by="Datumtijd")
        df2["Diepte (m)"] = pd.to_numeric(df2["Diepte (m)"], errors="coerce") * -1
        df2["Chloriniteit (mg/l)"] = pd.to_numeric(df2["Chloriniteit (mg/l)"], errors="coerce")
        df2["Datum"] = parse_mixed_datetime(df2["Datum"])
        df2 = df2.dropna(subset=["Diepte (m)", "Chloriniteit (mg/l)"])
        if df2.empty:
            continue

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
