"""
Visualization package entry point with subcommands.

Usage:
    python -m visualization robustness DATA_PATH --suffix SUFFIX
    python -m visualization robustness_quality DATA_PATH --pattern PATTERN
    python -m visualization watermark_quality DATA_PATH
"""

import argparse
import os
import sys

from visualization.utils import setup_plot_style, extract_suffixes


def main():
    parser = argparse.ArgumentParser(
        description='SemStamp visualization tools',
        usage='python -m visualization <command> [options]',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- robustness ---
    rob = subparsers.add_parser(
        'robustness',
        help='Fixed vs context-dependent comparison for a single paraphraser',
    )
    rob.add_argument('data_path', type=str, help='Base path to data directories')
    rob.add_argument('--suffix', type=str, default='-generated-parrot-bigram=False-threshold=0.0',
                     help='Suffix for data directories')
    rob.add_argument('--output', type=str, default=None, help='Output directory')

    # --- robustness_quality ---
    rq = subparsers.add_parser(
        'robustness_quality',
        help='Cross-paraphraser robustness vs quality analysis',
    )
    rq.add_argument('data_path', type=str, help='Base path to data directories')
    rq.add_argument('--pattern', type=str, default='*-generated-*',
                    help='Glob pattern for matching subdirectories')
    rq.add_argument('--output', type=str, default=None, help='Output directory')
    rq.add_argument('--force', action='store_true', help='Re-generate even if output exists')

    # --- watermark_quality ---
    wq = subparsers.add_parser(
        'watermark_quality',
        help='Watermark quality vs detectability (no paraphrasing)',
    )
    wq.add_argument('data_path', type=str, help='Base data directory')
    wq.add_argument('--output', type=str, default=None, help='Output directory')

    args = parser.parse_args()
    setup_plot_style()

    if args.command == 'robustness':
        from visualization.robustness import run_visualization
        run_visualization(args.data_path, args.suffix, args.output)

    elif args.command == 'robustness_quality':
        from visualization.robustness_quality import run_robustness_quality_visualization

        suffixes = extract_suffixes(args.data_path, args.pattern)
        if not suffixes:
            print(f"No matching directories found in {args.data_path} with pattern '{args.pattern}'")
            sys.exit(1)

        print(f"Found {len(suffixes)} unique paraphraser suffix(es):")
        for s in suffixes:
            print(f"  {s}")
        print()

        output_dir = args.output or os.path.join(args.data_path, 'figures', 'robustness_quality')
        if not args.force and os.path.exists(output_dir):
            print(f"Output directory already exists: {output_dir}")
            print("Use --force to re-generate.")
            sys.exit(0)

        run_robustness_quality_visualization(args.data_path, suffixes, output_dir)

    elif args.command == 'watermark_quality':
        from visualization.watermark_quality import run_watermark_quality_visualization
        run_watermark_quality_visualization(args.data_path, args.output)


if __name__ == '__main__':
    main()
