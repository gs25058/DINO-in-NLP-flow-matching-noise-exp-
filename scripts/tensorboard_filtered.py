#!/usr/bin/env python3
"""Mirror selected scalar events into a separate, live TensorBoard logdir."""
import os
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'results/tensorboard'
DEST = ROOT / 'results/tensorboard_filtered'
TAGS = {'eval/sts_b_dev_spearman', 'eval/alignment', 'eval/uniformity',
        'eval/effective_rank', 'train/KL_pt_ps', 'train/weighted_mean',
        'train/weighted_cov'}


def main():
    # A new output directory per process avoids overwriting prior event files.
    session = str(os.getpid())
    readers, writers, offsets = {}, {}, {}
    while True:
        for root, _, files in os.walk(SOURCE, followlinks=True):
            if not any(name.startswith('events.out.tfevents') for name in files):
                continue
            path = Path(root)
            try:
                reader = readers.setdefault(root, EventAccumulator(root, size_guidance={'scalars': 0}))
                reader.Reload()
                tags = TAGS.intersection(reader.Tags()['scalars'])
                if not tags:
                    continue
                if root not in writers:
                    out = DEST / path.relative_to(SOURCE) / session
                    out.mkdir(parents=True, exist_ok=True)
                    writers[root] = EventFileWriter(str(out))
                writer = writers[root]
                for tag in tags:
                    values = reader.Scalars(tag)
                    key = (root, tag)
                    for item in values[offsets.get(key, 0):]:
                        writer.add_event(Event(wall_time=item.wall_time, step=item.step,
                            summary=Summary(value=[Summary.Value(tag=tag, simple_value=item.value)])))
                    offsets[key] = len(values)
                writer.flush()
            except (OSError, ValueError) as exc:
                print(f'Unable to read {root}: {exc}', flush=True)
        time.sleep(10)


if __name__ == '__main__':
    main()
