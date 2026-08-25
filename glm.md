# Prompt สำหรับ Claude: AI Space Weather Forecasting PoC

## Context
Act as an expert AI Engineer and Space Physics Researcher. I am participating in the Intel AI Global Impact Festival and I have only 2 days left to submit my project. 

My project goal: "Space Weather Forecasting" - Using Machine Learning to predict Solar Storms (M-Class flares) to protect Earth's infrastructure (Power grids, GPS, Aviation) aligning with UN SDG 9 and 13.

## Problem
The NOAA API URL "https://services.swpc.noaa.gov/json/goes/primary/dxr-30-days.json" is broken. It returns an HTML directory listing instead of JSON, causing a JSONDecodeError.

## Requirements for the Code
Please write a complete, single-file Python script that acts as a Proof of Concept (PoC) for this project. 

1. **Environment Constraint**: I am using Python 3.13, so DO NOT use `scikit-learn-intelex` as it causes installation errors. Use standard `scikit-learn`.
2. **Data Sourcing**: Try to fetch real-time 30-day solar X-ray flux data from the NOAA SWPC API: "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-day.json"
3. **Fallback Mechanism**: Use a robust `try-except` block that catches `JSONDecodeError`. If the request fails or returns an HTML directory listing, automatically generate realistic simulated "Space Weather" data (base flux around 1e-6 to 5e-6, and solar storms around 1e-5 to 8e-5) for 30 days (720 hours) so the script NEVER breaks.
4. **Feature Engineering**: Shift the `long_flux` data by 1 to 5 hours to create lag features (`flux_prev_1` to `flux_prev_5`).
5. **Target Variable**: Create a binary classification `is_storm` (1 if `long_flux` > 1e-5, else 0).
6. **AI Model**: Train a `RandomForestClassifier` (standard scikit-learn) on an 80/20 train-test split.
7. **Outputs**: 
   - Print the Accuracy score and Classification Report.
   - Plot a matplotlib graph showing the X-ray flux over time with a red dashed line indicating the 1e-5 storm threshold.
   - Save the plot as `solar_storm_graph.png`.
8. **Intel Tie-in Comments**: Add comments in the code explaining how this model would be optimized using Intel OpenVINO for real-time Edge deployment at ground stations (this is for my pitch deck).

Please generate the full code, including imports, so I can copy-paste and run it immediately.