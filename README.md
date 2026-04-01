# DS 4320 Project 1: Wildfires
### Executive Summary
This repository contains all materials for a data analysis project on wildfire risk using the UCI Forest Fires dataset. It includes the raw and processed data, a normalized relational database structure, and SQL queries used to prepare a modeling dataset. The repository also contains the full data pipeline implemented in Python and DuckDB, along with a Jupyter Notebook documenting the data cleaning, transformation, and analysis steps. Additionally, it includes a press release for a general audience, supporting documentation such as domain exposition and data dictionaries, and a license and DOI for reproducibility and citation.

### Zoe Hendershot

### njd5rd

### DOI
[![DOI](https://zenodo.org/badge/1195707301.svg)](https://doi.org/10.5281/zenodo.19363213)

### Press release - link
[Which Weather Conditions Lead to the Most Destructive Wildfires](./pressrelease.md)
### link to data folder
[Data Folder](https://myuva-my.sharepoint.com/:f:/r/personal/njd5rd_virginia_edu/Documents/Data%20Design%20Project%201/Project%20Data?csf=1&web=1&e=xwJJey)
### link to pipeline files
[Data Pipeline Markdown File](https://myuva-my.sharepoint.com/:t:/r/personal/njd5rd_virginia_edu/Documents/Data%20Design%20Project%201/Pipeline%20Files/datapipeline.md?csf=1&web=1&e=Ghl3sU)
[Data Pipeline Jupyter Notebook File](https://myuva-my.sharepoint.com/:u:/r/personal/njd5rd_virginia_edu/Documents/Data%20Design%20Project%201/Pipeline%20Files/datapipeline.ipynb?csf=1&web=1&e=haJye2)

### License
This project is licensed under the MIT License.  
See the LICENSE file for details:  
[LICENSE](https://github.com/zoehendershot/Data-design-project-1/blob/main/LICENSE)

## Problem Definition
## State the initial general problem and refined specific problem statement
Initial General Problem: Wildfires are a growing environmental hazard that threaten ecosystems, human populations, and air quality, making it important to understand the conditions that influence their severity.

Refined Specific Problem: This project investigates how variation in environmental and meteorological conditions, specifically temperature, relative humidity, wind speed, rainfall, and drought-related fire weather indices, are associated with differences in wildfire severity, measured by burned area. It aims to identify which combinations of these conditions are most strongly linked to larger, more severe wildfire events, and to better understand how these factors interact to influence the scale of fires.

Refinement Rationale: The initial problem of understanding wildfire behavior is broad and difficult to analyze directly, as wildfire severity is influenced by many complex and often unobserved factors. By refining the problem to focus specifically on how environmental and meteorological conditions relate to burned area, the analysis becomes more targeted and measurable. Burned area provides a continuous indicator of wildfire severity, allowing for meaningful comparisons across events. Additionally, narrowing the focus to variables such as temperature, humidity, wind, rainfall, and drought-related indices leverages the available data while concentrating on factors that are theoretically linked to fire spread. This refinement allows the analysis to move beyond general description and instead examine how variation in these conditions corresponds to differences in wildfire severity.

Project Motivation: Understanding the environmental conditions associated with more severe wildfires is important for improving risk assessment and informing prevention and response strategies. As climate patterns shift and extreme weather events become more frequent, identifying how factors such as temperature, humidity, wind, and drought conditions relate to wildfire severity can help highlight situations where fires are more likely to grow large and cause greater damage. Beyond its environmental relevance, this project also demonstrates how data can be used to study complex, real-world phenomena, where outcomes are influenced by many interacting factors and may not be easily predictable. By analyzing these relationships, the project contributes to a broader effort to better understand and manage wildfire risk.

## Headline of Press Release and link to separate markdown file containing the press release
[Which Weather Conditions Lead to the Most Destructive Wildfires](./pressrelease.md)

## Domain exposition                          
## Terminology- table summarizing jargon, KPIs, etc. 
| Term                           | Definition                                                                                       | Form of Measurement / KPI                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Wildfire                       | An uncontrolled fire occurring in natural vegetation such as forests, grasslands, or shrublands. | Observed fire events recorded.                       |
| Burned Area                    | The total amount of land affected by a wildfire.                                                 | Measured in hectares; used as the primary outcome variable in wildfire severity analysis. |
| FFMC (Fine Fuel Moisture Code) | Indicator of moisture content in surface litter and fine fuels.                                  | Numerical index representing fuel dryness and ignition potential.                         |
| DMC (Duff Moisture Code)       | Measure of moisture in moderately deep organic layers of soil.                                   | Numerical drought indicator.                                   |
| DC (Drought Code)              | Long-term indicator of deep soil moisture and drought conditions.                                | Numerical index used to estimate long-term fire risk.                                     |
| ISI (Initial Spread Index)     | Indicator of how quickly a fire may spread immediately after ignition.                           | Derived from wind speed and fuel moisture.                                                |
| Relative Humidity              | The amount of moisture present in the air.                                                       | Percentage measurement collected from weather data.                                       |
| Wind Speed                     | The speed of wind that can influence fire spread.                                          | Measured in km/h from meteorological data.                                                |
| Fire Weather Index System      | A widely used model that estimates wildfire danger using weather and fuel moisture indicators.   | Calculated from weather variables and fuel moisture codes.                                |

## Paragraph explaining the domain the project lives in
This project lies within the domain of environmental data science and wildfire prediction. Wildfires are complex natural events influenced by a wide range of environmental conditions. Researchers and environmental agencies use meteorological and ecological data to better understand the conditions that contribute to wildfire occurrence and severity. The Forest Fires dataset from the UCI Machine Learning Repository contains records of wildfire events in Montesinho Natural Park in Portugal along with weather conditions and drought indicators collected at the time of each fire. By analyzing relationships between environmental variables and burned area, researchers can identify patterns that may help improve wildfire risk assessment and environmental management strategies.

## Background reading Separate folder 
[Wildfires Background Reading Folder](https://myuva-my.sharepoint.com/:f:/g/personal/njd5rd_virginia_edu/IgBhyR_FLbvzR6G51Pql1TSYAfX7vFkfYp4B1Z-r_BDv9G4?e=Qnxodo)

## Table - showing a summary of the readings, one row per item, includes title, brief description, and link to file in folder
| Title                                                                                            | Description                                                                                                                                        | File                                                   |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Prediction of Area Burned by Forest Fires Using Machine Learning Algorithms (Bhat et al., 2022)  | Research paper exploring how machine learning models can be used to predict wildfire burned area using environmental and meteorological variables. | `wildfires_background_reading/Bhat Forest Fire Prediction.url`  |
| Climate Change and Wildfire in Idaho, Oregon, and Washington                                                                    | Article explaining how climate change, rising temperatures, and drought conditions influence wildfire frequency and severity.                      | `wildfires_background_reading/Climate Change and Wildfires.url` |
| A Data Mining Approach to Predict Forest Fires Using Meteorological Data (Cortez & Morais, 2007) | Original research paper introducing the Forest Fires dataset and analyzing how meteorological variables can help predict wildfire severity.        | `wildfires_background_reading/Cortez Morais 2007.url`           |
| Fire Weather Index (FWI) System                                                                  | Documentation describing the Fire Weather Index system used to estimate wildfire danger using weather conditions and fuel moisture levels.         | `wildfires_background_reading/Fire Weather Index System.url`    |
| Forest Fires Dataset: UCI Machine Learning Repository                                           | Official dataset documentation explaining the variables, structure, and collection process for the wildfire dataset used in this project.          | `wildfires_background_reading/Forest Fires UCI ML Repo.url`     |

## Data Creation
## Paragraph or two explaining the raw data acquisition process (provenance).
The dataset used in my analysis is from the UCI Machine Learning Repository, specifically the Forest Fires dataset. This dataset contains observations of wildfire events in a forested region in Northeast Portugal, and includes spatial coordinates, temporal variables (month and day), environmental conditions (temperature, humidity, wind, and rain), and fire-related indices (FFMC, DMC, DC, ISI), along with the burned area. The data was downloaded from the UCI repository as a compressed ZIP file, which was then unzipped to extract the CSV file. After loading the dataset, I examined its structure briefly through excel and found that the dataset was already well-structured, limiting the need for any major preprocessing. The final dataset is organized as a single table, where each row represents an observation of wildfire conditions and the associated burned area.

## Code Table showing the code used to create the data, one row per file, with a brief description and link to code in repo

| File Name             | Description                                                                 | Link |
|-----------------------|-----------------------------------------------------------------------------|------|
| loaddata.ipynb  | Loads the UCI Forest Fires dataset from CSV, performs basic cleaning, derives 2 new features for analysis, prints a brief summary of the dataset structure and missing values, and conducts a short analysis for verification. | [sample code](https://colab.research.google.com/drive/1mtre0p96ivqYwk1kIvuhLl5D5OhJ2grv?usp=sharing) |

## Bias Identification description of how bias could be/was introduced in the data collection process
Bias may be introduced into the Forest Fires dataset through several aspects of the data collection process. First, the dataset is limited to a specific geographic region (the Montesinho Natural Park in Portugal), which introduces a form of sampling bias. The environmental and wildfire patterns observed may not generalize to other regions with different climates or vegetation.

Second, the dataset uses spatial coordinates (X and Y) based on a grid system rather than precise geographic latitude and longitude. This basically pixelates the data, where instead of capturing the exact location of each fire, observations are grouped into grid cells. This could possibly lead to multiple fires occurring in different parts of the same grid cell being treated as a single observation, which results in a loss of detail and obscurity of local variation.

Lastly, there may be survivorship bias, as the dataset likely includes only recorded wildfire events. Smaller or less significant fires may be underrepresented, while larger or more impactful fires are more likely to be captured in the data.

## Bias Mitigation description of how biases can be handled/quantified/accounted for in analysis
Each type of bias mentioned requires its own unique solution. To account for geographic sampling bias, conclusions should be limited to the Montesinho Natural Park region (or regions with very similar geographic features/climate) rather than generalized broadly. To generalize further, we would have to collect more data from other regions.

To mitigate spatial resolution bias introduced by the grid-based (X and Y) system, analysis should focus on aggregated trends rather than precise location-based conclusions. For example, grouping observations by larger patterns (such as overall fire frequency or average burned area) helps reduce the impact of the “pixelation” effect and avoids overinterpreting small spatial differences.

Finally, survivorship bias can be accounted for by acknowledging that the dataset may underrepresent smaller fires. Analyses should avoid assuming that the dataset reflects all wildfire activity and instead interpret results as representative of recorded events, which may skew toward larger or more easily detected fires. This is an example of bias that would have to be documented as a source of uncertainty.

## Rationale for critical decisions, especially judgement calls, and places that can introduce/mitigate uncertainty
Several key decisions were made in selecting the Forest Fires dataset for my analysis. I chose to use the dataset without attempting to alter or correct inherent limitations. Certain characteristics are fundamental to how the data was collected and cannot be meaningfully reconstructed at a finer level of detail without redesign the collecting process and starting from scratch.

The survivorship bias specifically presents a limitation that cannot be directly corrected. Rather than attempting to adjust for this, the approach I have decided to take is to explicitly acknowledge this limitation and interpret results as reflecting observed and recorded fires, rather than all fire activity. Transparency is key to moving forward even when the data may have inherent bias.

Similarly, the use of grid-based coordinates introduces uncertainty by obscuring detailed spatial variation. Because of this, I made the decision to focus on broader patterns and relationships in the data, which are more robust to this type of spatial uncertainty. Overall, the primary strategy for handling uncertainty in this project is not to eliminate it, but to clearly document, acknowledge, and account for it in the interpretation of results.

## Metadata

## Schema ER diagram at the logical level 
```
+--------------------------------------------------+
|               forestfiresdataset                |
+--------------------------------------------------+
| pk record_id                                     |
|    x                                             |
|    y                                             |
|    month                                         |
|    day                                           |
|    season                                        |
|    ffmc                                          |
|    dmc                                           |
|    dc                                            |
|    isi                                           |
|    temp                                          |
|    rh                                            |
|    wind                                          |
|    rain                                          |
|    area                                          |
|    log_area                                      |
+--------------------------------------------------+
```

## Data Table listing all of the tables in the dataset, one line per table, brief description and link to csv file

The dataset was normalized into relational tables to reduce redundancy and improve organization, with each table representing a distinct component of wildfire data. The original dataset is also included for reference.

| Table Name | Description | Link |
|-----------|------------|------|
| `forestfires` | Original flat dataset containing all wildfire observations and variables prior to normalization | [forestfires.csv](link) |
| `fires` | Main fact table containing one record per wildfire event, including foreign keys linking to location, time, weather, and fire index tables, along with burned area and log-transformed area | [fires.csv](link) |
| `locations` | Lookup table containing spatial grid coordinates (`x`, `y`) for each unique fire location | [locations.csv](link) |
| `time_info` | Lookup table containing temporal information for each fire event, including month, day, and derived season | [time_info.csv](link) |
| `weather` | Lookup table containing meteorological conditions such as temperature, relative humidity, wind speed, and rainfall | [weather.csv](link) |
| `fire_indices` | Lookup table containing fire weather index variables (FFMC, DMC, DC, ISI) that measure fuel and drought conditions | [fire_indices.csv](link) |

## Data Dictionary Table with one row per feature in the data set containing: name, data type, description, example

| Variable Name | Data Type | Description                                                                 | Example |
|---------------|----------|-----------------------------------------------------------------------------|---------|
| record_id     | int      | Unique identifier for each wildfire observation                             | 1       |
| x             | int      | x-axis spatial coordinate within Montesinho park grid (1–9)                | 7       |
| y             | int      | y-axis spatial coordinate within Montesinho park grid (2–9)                | 5       |
| month         | string   | Month of the year ('jan' to 'dec')                                          | aug     |
| day           | string   | Day of the week ('mon' to 'sun')                                            | fri     |
| season        | string   | Season derived from month                                                   | summer  |
| ffmc          | float    | Fine Fuel Moisture Code: measures moisture of surface litter and fine fuels; higher values indicate drier, more flammable conditions | 85.2    |
| dmc           | float    | Duff Moisture Code: represents moisture in moderately deep organic layers; higher values indicate sustained dryness | 26.2    |
| dc            | float    | Drought Code: reflects long-term moisture conditions in deep soil layers; higher values indicate severe drought | 94.3    |
| isi           | float    | Initial Spread Index: estimates expected rate of fire spread based on wind and fuel conditions | 5.1     |
| temp          | float    | Temperature in degrees Celsius                                              | 18.0    |
| rh            | int      | Relative humidity (%), indicating moisture in the air                       | 45      |
| wind          | float    | Wind speed, influencing fire spread rate                                    | 4.0     |
| rain          | float    | Rainfall amount, representing recent precipitation                          | 0.0     |
| area          | float    | Burned area of the forest (hectares)                                       | 12.5    |
| log_area      | float    | Log-transformed burned area to reduce right skew and stabilize variance     | 2.6     |


## Data Dictionary quantification of uncertainty for numerical features

| Variable | Data Type | Uncertainty Metric |
|----------|----------|--------------------------------|
| isi      | float    | Standard deviation (σ ≈ 4.58) and coefficient of variation (CV ≈ 0.51), indicating moderate variability in fire spread potential relative to its average level |
| temp     | float    | Measurement uncertainty ±0.5°C and observed variability (σ ≈ 5.83°C), showing typical natural fluctuations in temperature |
| area     | float    | Highly right-skewed (skewness ≈ 12.8) with large variability (σ ≈ 63.89) and IQR ≈ 6.57, indicating most fires are small but a few extreme fires create high uncertainty |
| log_area | float    | Lower variability (σ ≈ 1.4) and reduced skewness (≈ 1.22), showing that the log transformation makes wildfire severity more stable and easier to analyze |






