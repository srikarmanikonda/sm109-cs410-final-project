import requests
import pandas as pd
import time


BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

def fetch_trials(limit=100):
    """
    Fetches recent clinical trials from ClinicalTrials.gov API.
    """
    params = {
        "format": "json",
        "pageSize": limit,
    }

    print(f"Fetching {limit} trials from {BASE_URL}...")
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        studies = data.get('studies', [])
        processed_data = []

        for study in studies:
            protocol = study.get('protocolSection', {})
            id_module = protocol.get('identificationModule', {})
            status_module = protocol.get('statusModule', {})
            design_module = protocol.get('designModule', {})
            conditions_module = protocol.get('conditionsModule', {})
            contacts_locations = protocol.get('contactsLocationsModule', {})

            nct_id = id_module.get('nctId', '')
            title = id_module.get('briefTitle', '')


            summary = protocol.get('descriptionModule', {}).get('briefSummary', '')

            conditions = conditions_module.get('conditions', [])
            condition_str = ", ".join(conditions) if conditions else ""

            phases = design_module.get('phases', [])
            phase_str = ", ".join(phases) if phases else "Not Applicable"

            status = status_module.get('overallStatus', '')

            locations = contacts_locations.get('locations', [])
            city = ""
            state = ""

            if locations:
                us_loc = next((loc for loc in locations if loc.get('country') == 'United States'), None)
                target_loc = us_loc if us_loc else locations[0]

                city = target_loc.get('city', '')
                state = target_loc.get('state', '')

            processed_data.append({
                'NCTId': nct_id,
                'BriefTitle': title,
                'BriefSummary': summary,
                'Condition': condition_str,
                'Phase': phase_str,
                'OverallStatus': status,
                'LocationCity': city,
                'LocationState': state
            })

        return pd.DataFrame(processed_data)

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = fetch_trials(limit=100)
    if not df.empty:
        output_path = "data/sample_trials.csv"
        df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(df)} trials to {output_path}")
        print(df.head())
    else:
        print("No data found or error occurred.")

