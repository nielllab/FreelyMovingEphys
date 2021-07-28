# FreelyMovingEphys Documentation

## Navigating documentation
A setup and instillation walkthrough can be found in the [installation guide](installation.md).

An overview of each module, a description of the full workflow from raw data to figures, and a quick tutorial for running the analysis can be found in the [user guide](user_guide.md).

For the required format of data, metadata, and directory structures, see the [data formatting guide](data_formatting.md).

Details on the main functions user are likley to interact with can be found in the [function details guide](function_details.md) and the guide for other functions can be found in [this](additional_functions.md) guide.

## Typical sequence (leading up to specific project analysis)
1. Preprocess ephys data and merge recordings
2. Kilosort
3. Phy2
4. Split ephys recordings
5. Write yaml config or batch csv
6. Run either session or batch analysis
7. Run additional project-specific analysis
