import glob
import os
import re
import pandas as pd
import plotly.express as px

# Excel-bron (ondersteunt zowel een specifiek bestand als een directory met Excelbestanden)
TARGET_FILE = os.path.join(
    ".", "data", "metingen"
)
# Nieuwe bronmap voor losse CSV-bestanden per meetpunt
CSV_INPUT_DIR = os.path.join(".", "data", "metingen", "csv")
# Overzichtssheet overslaan; dit is geen meetsheet
SKIP_SHEETS = {"IJsselmeer", "Overzicht_Metingen", "QC", "Detectie", "Instellingen"}
# Bestaande CSV behouden en nieuwe metingen eraan toevoegen
KEEP_EXISTING_CSV = True
CSV_PATH = os.path.join("data", "chloridemetingen ijsselmeer.csv")
# Maximale RD-afstand (meters) voor fallback op coördinaten als bestandsnaam niet matcht
LOCATION_COORD_TOLERANCE = 500

# Labels uit het voorblad/overzicht die geen meetmetadata zijn en daarom
# nooit als kolom of losse rij in de uitvoer-CSV mogen terechtkomen.
UNWANTED_OVERVIEW_LABELS = {
    "Zoutmetingen IGL bv i.o.v. Rijkswaterstaat Centrale Informatievoorziening",
    "Meetpunt",
    "Ronde",
    "Datum_x",
    "Starttijd (UTC+1)",
    "x voorgeschreven (RD)",
    "y voorgeschreven (RD)",
    "x werkelijk (RD)",
    "y werkelijk (RD)",
    "Datum_y",
}
UNWANTED_OVERVIEW_KEYS = {
    re.sub(r"[^a-z0-9]+", "", label.strip().lower())
    for label in UNWANTED_OVERVIEW_LABELS
}


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


def path_contains_verwerkt(path):
    """Controleer of een pad in een map 'verwerkt' staat (case-insensitive)."""
    normalized = os.path.normpath(path)
    parts = [part.lower() for part in normalized.split(os.sep) if part]
    return "verwerkt" in parts


def list_excel_files(target_path):
    """Geef alle Excelbronnen terug, behalve bestanden onder een map 'verwerkt'.

    Ondersteunt zowel een expliciet bestandspad als een directory.
    """
    if not target_path:
        return []

    if os.path.isfile(target_path):
        _, ext = os.path.splitext(target_path)
        if ext.lower() in {".xlsx", ".xls"} and not path_contains_verwerkt(target_path):
            return [target_path]
        return []

    if not os.path.isdir(target_path):
        return []

    pattern_xlsx = os.path.join(target_path, "**", "*.xlsx")
    pattern_xls = os.path.join(target_path, "**", "*.xls")
    files = sorted(
        set(glob.glob(pattern_xlsx, recursive=True) + glob.glob(pattern_xls, recursive=True))
    )
    return [file_path for file_path in files if not path_contains_verwerkt(file_path)]


def list_csv_files(csv_dir):
    """Geef alle losse CSV-bronbestanden terug, behalve onder een map 'verwerkt'."""
    if not csv_dir or not os.path.isdir(csv_dir):
        return []

    files = sorted(glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True))
    return [file_path for file_path in files if not path_contains_verwerkt(file_path)]


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

    Bestanden onder een map 'verwerkt' worden genegeerd.
    """
    mapping_files = list_excel_files(csv_dir)
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
            mapping["filebasename"] = (
                df[basename_col]
                .astype(str)
                .str.strip()
                .str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
            )
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


def find_measurement_header_row(xlsx, sheet):
    """Zoek de Excelrij waarop de meettabel begint."""

    preview = pd.read_excel(
        xlsx,
        sheet_name=sheet,
        header=None,
        nrows=25,
    )

    expected_terms = {
        "diepte",
        "temperatuur",
        "geleidendheid",
        "chloriniteit",
        "datum",
        "tijd",
    }

    for row_index, row in preview.iterrows():
        values = {
            str(value).strip().lower()
            for value in row.dropna()
        }

        matches = 0

        for term in expected_terms:
            if any(term in value for value in values):
                matches += 1

        # Minimaal drie bekende meetkolommen op dezelfde rij
        if matches >= 3:
            return row_index

    raise ValueError(
        f"Geen herkenbare kopregel voor meetgegevens gevonden "
        f"in sheet '{sheet}'."
    )

def extract_data_from_sheet(xlsx, sheet):
    """Lees een profielsheet en zet deze om naar het vaste uitvoerformaat."""
    header_row = find_measurement_header_row(xlsx, sheet)

    # Metadata zijn de parameter/waarde-paren boven de meettabel.
    meta_raw = pd.read_excel(
        xlsx, sheet_name=sheet, header=None, usecols="A:B", nrows=header_row
    )
    metadata = {}
    for parameter, value in meta_raw.itertuples(index=False, name=None):
        if pd.isna(parameter):
            continue
        key = normalize_column_name(parameter)
        if not key or key.startswith("zoutmetingen"):
            continue
        metadata[key] = value

    # Gebruik de werkelijk gevonden header, niet een vast rijnummer.
    data = pd.read_excel(xlsx, sheet_name=sheet, header=header_row)
    data = data.dropna(how="all").dropna(axis=1, how="all")

    depth_col = find_column(data.columns, ["Diepte", "Diepte (m)", "depth"])
    temp_col = find_column(data.columns, ["Temperatuur", "Temperatuur (graden Celsius)", "temperature"])
    cond_col = find_column(data.columns, ["Geleidendheid", "Geleidendheid (mS/cm)", "conductivity", "ec"])
    cl_col = find_column(data.columns, ["Chloriniteit", "Chloriniteit (mg/l)", "chloride", "cl"])
    dt_col = find_column(data.columns, ["Datum/Tijd UTC+1", "Datum/Tijd UTC", "Datumtijd", "datetime", "timestamp"])
    date_col = find_column(data.columns, ["Datum"])
    time_col = find_column(data.columns, ["Tijd (UTC)", "Tijd (UTC+1)", "Tijd"])

    required = {"diepte": depth_col, "temperatuur": temp_col, "geleidendheid": cond_col, "chloriniteit": cl_col}
    missing = [name for name, column in required.items() if column is None]
    if missing or (dt_col is None and (date_col is None or time_col is None)):
        raise ValueError(
            f"Sheet '{sheet}' mist verplichte meetkolommen: {missing}. "
            f"Ingelezen kolommen: {list(data.columns)}"
        )

    output = pd.DataFrame(index=data.index)
    output["Diepte (m)"] = parse_numeric_series(data[depth_col])
    output["Temperatuur (graden Celsius)"] = parse_numeric_series(data[temp_col])
    output["Geleidendheid (mS/cm)"] = parse_numeric_series(data[cond_col])
    output["Chloriniteit (mg/l)"] = parse_numeric_series(data[cl_col])

    if dt_col is not None:
        raw_dt = data[dt_col]
        numeric_dt = pd.to_numeric(raw_dt, errors="coerce")
        output["Datumtijd"] = pd.NaT
        numeric_mask = numeric_dt.notna()
        if numeric_mask.any():
            output.loc[numeric_mask, "Datumtijd"] = pd.to_datetime(
                numeric_dt[numeric_mask], unit="D", origin="1899-12-30", errors="coerce"
            )
        text_mask = output["Datumtijd"].isna()
        if text_mask.any():
            output.loc[text_mask, "Datumtijd"] = parse_mixed_datetime(raw_dt.loc[text_mask])
    else:
        dates = parse_mixed_datetime(data[date_col]).dt.normalize()
        times = pd.to_timedelta(
            data[time_col].astype(str).str.replace(" UTC", "", regex=False).str.strip(),
            errors="coerce",
        )
        output["Datumtijd"] = dates + times

    # De eenhedenregel heeft geen numerieke diepte/chloriniteit en geen datumtijd.
    output = output.dropna(subset=["Diepte (m)", "Chloriniteit (mg/l)", "Datumtijd"])
    if output.empty:
        return output

    output["Datum"] = output["Datumtijd"].dt.normalize()
    output["Tijd (UTC)"] = output["Datumtijd"].dt.strftime("%H:%M:%S")

    meetpunt = metadata.get("meetpunt")
    ronde = metadata.get("ronde")
    profiel_id = metadata.get("id")
    locatie = metadata.get("locatie")
    x_actual = metadata.get("xwerkelijkrd")
    y_actual = metadata.get("ywerkelijkrd")
    x_prescribed = metadata.get("xvoorgeschrevenrd")
    y_prescribed = metadata.get("yvoorgeschrevenrd")

    output["GPS-Mark"] = profiel_id
    output["ID"] = profiel_id
    output["Locatie"] = locatie
    # In nieuwe bestanden is Ronde leeg; Meetpunt bevat de profielnaam.
    output["Rondnr"] = ronde if pd.notna(ronde) else meetpunt
    output["x-coordinaat (RD)"] = x_actual if pd.notna(x_actual) else x_prescribed
    output["y-coordinaat (RD)"] = y_actual if pd.notna(y_actual) else y_prescribed
    output["Maximale diepte [m]"] = output["Diepte (m)"].max()

    column_order = [
        "GPS-Mark", "ID", "Locatie", "Rondnr", "x-coordinaat (RD)",
        "y-coordinaat (RD)", "Maximale diepte [m]", "Diepte (m)",
        "Temperatuur (graden Celsius)", "Geleidendheid (mS/cm)",
        "Chloriniteit (mg/l)", "Datumtijd", "Datum", "Tijd (UTC)"
    ]
    return output[column_order]


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
        df_existing = pd.read_csv(CSV_PATH, dtype=str, low_memory=False)
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



def load_excel_overview_mapping(xlsx):
    """Lees Overzicht_Metingen als aanvullende controle/mapping."""
    if "Overzicht_Metingen" not in xlsx.sheet_names:
        return pd.DataFrame()
    overview = pd.read_excel(xlsx, sheet_name="Overzicht_Metingen")
    overview = overview.dropna(how="all")
    if overview.empty:
        return overview
    overview.columns = [str(column).strip() for column in overview.columns]
    return overview


def enrich_excel_measurements(df, sheet, overview_mapping):
    """Vul alleen werkelijk ontbrekende profielvelden aan uit Overzicht_Metingen."""
    result = df.copy()
    if overview_mapping is None or overview_mapping.empty:
        return result

    id_match = re.match(r"^(\d+)_", str(sheet))
    if not id_match:
        return result
    profile_id = int(id_match.group(1))
    id_col = find_column(overview_mapping.columns, ["ID"])
    if id_col is None:
        return result
    ids = pd.to_numeric(overview_mapping[id_col], errors="coerce")
    selected = overview_mapping.loc[ids.eq(profile_id)]
    if selected.empty:
        return result
    record = selected.iloc[0]

    source_map = {
        "ID": ["ID"],
        "Locatie": ["Locatie"],
        "Rondnr": ["Ronde", "Meetpunt"],
        "x-coordinaat (RD)": ["X werkelijk (m)", "X voorgeschreven (m)"],
        "y-coordinaat (RD)": ["Y werkelijk (m)", "Y voorgeschreven (m)"],
        "Maximale diepte [m]": ["Maximale diepte (m)"],
    }
    for target, candidates in source_map.items():
        value = pd.NA
        for candidate in candidates:
            source = find_column(overview_mapping.columns, [candidate])
            if source is not None and pd.notna(record[source]):
                value = record[source]
                break
        if pd.notna(value):
            if target not in result.columns:
                result[target] = value
            else:
                result[target] = result[target].fillna(value)

    if "ID" in result.columns:
        result["GPS-Mark"] = result.get("GPS-Mark", pd.Series(index=result.index, dtype=object)).fillna(result["ID"])
    return result


def collect_from_excel(target_file):
    df_compleet = pd.DataFrame()

    with pd.ExcelFile(target_file) as xlsx:
        overview_mapping = load_excel_overview_mapping(xlsx)
        for sheet in sorted(xlsx.sheet_names):
            if sheet in SKIP_SHEETS:
                print(f"Sheet overgeslagen volgens SKIP_SHEETS: {sheet}")
                continue

            try:
                df_single = extract_data_from_sheet(xlsx, sheet)
            except Exception as exc:
                print(
                    f"Sheet '{sheet}' overgeslagen wegens onverwachte "
                    f"indeling: {exc}"
                )
                continue

            if df_single.empty:
                print(f"Sheet '{sheet}' bevat geen bruikbare meetdata.")
                continue

            df_single = enrich_excel_measurements(df_single, sheet, overview_mapping)
            df_single["sheet"] = sheet
            df_single["filename"] = os.path.basename(target_file)

            df_compleet = pd.concat(
                [df_compleet, df_single],
                ignore_index=True
            )

    return df_compleet


def collect_from_excel_sources(excel_files):
    frames = []
    for excel_file in excel_files:
        try:
            df_single = collect_from_excel(excel_file)
        except Exception as exc:
            print(f"Excel overgeslagen wegens fout ({excel_file}): {exc}")
            continue
        if df_single.empty:
            continue
        frames.append(df_single)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def collect_from_csv_directory(csv_files, location_mapping=None):
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


def remove_overview_artifacts(df):
    """Verwijder voorbladlabels die per ongeluk als kolom of losse rij zijn ingelezen."""
    if df.empty:
        return df

    result = df.copy()

    # Verwijder ongewenste kolommen, ook bij kleine verschillen in hoofdletters
    # en leestekens.
    columns_to_drop = [
        column
        for column in result.columns
        if normalize_column_name(column) in UNWANTED_OVERVIEW_KEYS
    ]
    if columns_to_drop:
        result = result.drop(columns=columns_to_drop)
        print(f"Overzichtskolommen verwijderd: {columns_to_drop}")

    # Extra vangnet voor losse overzichtsregels: verwijder alleen zeer lege
    # regels waarin een van de bekende labels als celwaarde voorkomt.
    non_empty_count = result.notna().sum(axis=1)
    overview_row_mask = pd.Series(False, index=result.index)
    for column in result.columns:
        cell_keys = result[column].astype("string").fillna("").map(normalize_column_name)
        overview_row_mask |= cell_keys.isin(UNWANTED_OVERVIEW_KEYS)
    overview_row_mask &= non_empty_count <= 2

    if overview_row_mask.any():
        print(f"Losse overzichtsregels verwijderd: {int(overview_row_mask.sum())}")
        result = result.loc[~overview_row_mask].copy()

    return result


def build_measurement_csv():
    sources_found = False
    frames = []

    excel_files = list_excel_files(TARGET_FILE)
    csv_files = list_csv_files(CSV_INPUT_DIR)
    location_mapping = load_location_mapping(CSV_INPUT_DIR) if os.path.isdir(CSV_INPUT_DIR) else pd.DataFrame()

    if excel_files:
        sources_found = True
        frames.append(collect_from_excel_sources(excel_files))

    if csv_files:
        sources_found = True
        frames.append(collect_from_csv_directory(csv_files, location_mapping=location_mapping))

    if not sources_found:
        raise FileNotFoundError(
            f"Geen invoerbestanden gevonden. Verwacht Excel onder: {TARGET_FILE} of CSV-map: {CSV_INPUT_DIR}"
        )

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("Er is geen meetdata ingelezen uit de gevonden bronbestanden.")

    df_compleet = pd.concat(frames, ignore_index=True, sort=False)
    df_compleet = remove_overview_artifacts(df_compleet)

    #rename_dict = {
    #    "15_MG": "MG_15",
    #    "16_MG": "MG_16",
    #    "17_MG": "MG_17",
    #    "18_MG": "MG_18",
    #    "19_MG": "MG_19",
    #    "VG_DO_14,25": "VG_DO_14.25",
    #}
    #if "sheet" in df_compleet.columns:
    #    df_compleet = df_compleet.replace({"sheet": rename_dict})
    #if "Rondnr" in df_compleet.columns:
    #    df_compleet = df_compleet.replace({"Rondnr": rename_dict})

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
    df_output = remove_overview_artifacts(df_output)
    ensure_parent_dir(CSV_PATH)
    df_output.to_csv(CSV_PATH, index=False)
    print(f"CSV bijgewerkt: {CSV_PATH}")
    print(f"Aantal rijen in CSV: {len(df_output)}")


def get_visualisation_source_df(df):
    """Bepaal de brondata voor 2D-visualisaties.

    Regels:
    - Als er Excelbestanden gevonden zijn onder TARGET_FILE, gebruik dan alleen
      rijen uit die Excelbestanden.
    - Zo niet, gebruik de volledige samengestelde CSV zoals voorheen.
    - Bestanden onder een map 'verwerkt' zijn al uitgesloten bij het verzamelen.
    """
    excel_files = list_excel_files(TARGET_FILE)
    if not excel_files or "filename" not in df.columns:
        return df

    excel_basenames = {os.path.basename(path) for path in excel_files}
    df_excel_only = df[df["filename"].astype(str).isin(excel_basenames)].copy()
    if df_excel_only.empty:
        print(
            "Excelbron gevonden, maar geen overeenkomstige Excel-rijen in de samengestelde CSV. "
            "Val terug op volledige dataset voor visualisaties."
        )
        return df

    print(
        f"Excelbron gevonden ({len(excel_basenames)} bestand(en)); "
        "2D-visualisaties worden alleen op basis van Exceldata gemaakt."
    )
    return df_excel_only


def create_visualisations():
    if not os.path.exists(CSV_PATH):
        print(f"Geen CSV gevonden voor visualisaties: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH, dtype=str, low_memory=False)
    if df.empty or "sheet" not in df.columns:
        print("Geen data beschikbaar voor visualisaties.")
        return

    df["Datum"] = parse_mixed_datetime(df["Datum"])
    df["Datumtijd"] = parse_mixed_datetime(df["Datumtijd"])
    df = get_visualisation_source_df(df)
    if df.empty:
        print("Geen data beschikbaar voor visualisaties na bronfiltering.")
        return

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
