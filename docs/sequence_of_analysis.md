# FreelyMovingEphys - Sequence of Analysis

A typical sequence of analysis steps:

1. run Matlab script `preprocessEphysData.m`
2. Kilosort
3. Phy2
4. run python module `split_ephys_recordings` for an entire animal directory
5. run python module `project_analysis.map_receptive_fields` (*optional*) for an animal's white noise recording directory only
6. run python module `preprocessing` for entire animal directory
7. run python module `project_analysis.ephys` for each recording within an animal directory