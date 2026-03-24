from attention.sequence_builder import build_sequences, parse_args
from pathlib import Path


if __name__ == "__main__":
    args = parse_args()
    build_sequences(
        jsonl_path=Path(args.jsonl),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        min_track_len=args.min_track_len,
        max_track_len=args.max_track_len,
    )

