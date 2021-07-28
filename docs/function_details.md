# FreelyMovingEphys Function Details

## Ephys analysis


## Ephys population analysis
```
from utils.ephys import load_ephys
df = load_ephys(csv_filepath)
```
Returns a dataframe where each index is a unit and each column is a property saved out from ephys analysis from any of the recordings. Units are represented as an index only once, so that all recordings (e.g. fm1, hf1_wn, etc.) have all of their properties as columns for the unit shared across all recordings in the session.

```
from project_analysis.ephys.population_utils import make_session_summary
make_session_summary(df, /path/to/save/)
```
Saves a PDF summarizing each session, given `df`, a dataframe made by the function `load_ephys`.

```
from project_analysis.ephys.population_utils import make_unit_summary
unit_df = make_unit_summary(df, /path/to/save/)
```
Saves a PDF summarizing each unit in the dataframe created by the function `load_ephys`. Also returns a new version of the dataframe updated with new values (e.g. modulation index for tuning curves, etc.).