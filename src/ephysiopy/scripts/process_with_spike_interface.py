import argparse
from pathlib import Path

import spikeinterface.extractors as se
import spikeinterface.full as si

job_kwargs = {"progress_bar": True, "n_jobs": 8, "chunk_duration": "5s"}
si.set_global_job_kwargs(**job_kwargs)

my_pipeline = {
    "preprocessing": [
        {
            "name": "bandpass_filter",
            "params": {"freq_max": 3000, "freq_min": 300, "dtype": "float32"},
        },
        {
            "name": "common_reference",
            "params": {"operator": "median", "reference": "global"},
        },
        {"name": "phase_shift"},
        {"name": "whiten"},
        {"name": "detect_and_remove_bad_channels"},
        {
            "name": "detect_and_remove_artifacts",
            "params": {"recording_to_detect": "pipeline[raw]"},
        },
    ],
    "motion_correction": {
        "preset": "kilosort_like",
        # "chunk_duration": 10.0,
        # "max_translation_um": 50.0,
        # "max_rotation_deg": 5.0,
    },
    "sorting": {
        "sorter_name": "spykingcircus2",
        "verbose": True,
        "motion_correction": {"preset": "kilosort_like"},
    },
    "postprocessing": {
        "random_spikes": {},
        "noise_levels": {},
        "templates": {},
        "unit_locations": {"method": "center_of_mass"},
        "spike_amplitudes": {},
        "correlograms": {},
    },
}
parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
    "--recording_location",
    type=str,
    default="all",
    help="the path to the recording location",
    required=True,
)
parser.add_argument(
    "-s",
    "--stream_name",
    type=str,
    help="the name of the stream to process",
)
args = parser.parse_args()

assert Path(args.recording_location).exists(), (
    f"Recording location {args.recording_location} does not exist."
)

# get the available streams - first item in the list is the
# string names, second is the integer index of the stream
streams = se.get_neo_streams("openephysbinary", Path(args.recording_location))
options = streams[0]

for i, option in enumerate(options):
    print(f"{i}: {option}")

while True:
    choice = input(
        f"Select an option (0-{len(options) - 1}): ".format(len(options))
    ).strip()
    if choice.isdigit() and 0 <= int(choice) <= len(options) - 1:
        selected = options[int(choice)]
        print(f"You selected: {selected}")
        break
    print("Invalid selection, try again.")

rec = si.read_openephys(Path(args.recording_location), stream_name=selected)

preprocessed_rec = si.apply_preprocessing_pipeline(rec, my_pipeline["preprocessing"])

rec_corrected = si.correct_motion(preprocessed_rec, **my_pipeline["motion_correction"])

sorting = si.run_sorter(recording=preprocessed_rec, **my_pipeline["sorting"])

analyzer = si.create_sorting_analyzer(recording=preprocessed_rec, sorting=sorting)

analyzer.compute(my_pipeline["postprocessing"])
