#!/usr/bin/env python3
"""
Semantic Video Search - Main Entry Point

Usage:
    # Step 1: Ingest videos and create searchable index
    python -m sem_search.main ingest-all \\
        --data-dir /data/rosbags/processed/car \\
        --pattern "sess[789]/rosbag2_*/" \\
        --output-dir ./search_index

    # Step 2: Start the web server
    python -m sem_search.main serve --index-dir ./search_index --port 8000
"""
import os
os.environ['BLOSC_NTHREADS'] = '8'  # Optimize Blosc2 decompression

import argparse
import logging
import sys
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn

    from .serving.search_api import app, load_search_index
    
    # Load the index before starting
    index_path = Path(args.index_dir)
    if not index_path.exists():
        print(f"Error: Index directory not found: {args.index_dir}")
        print("Run 'ingest-all' first to create the index.")
        sys.exit(1)
    
    # Try to load the index
    try:
        load_search_index(str(index_path))
        logger.info(f"Loaded search index from {index_path}")
    except Exception as e:
        logger.warning(f"Could not load index: {e}")
        logger.warning("Search will use mock data")
    
    logger.info(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_query(args):
    """Standalone CLI search (no server): load the index locally, run the same
    ranking pipeline as the web UI, and print the top-N sequences with their time
    location + caption. Needs the Qwen3 query encoder (GPU fast / CPU slow)."""
    if args.device and args.device != 'auto':
        os.environ['SEMSEARCH_DEVICE'] = args.device   # encode_text_query honors this

    idx = Path(args.index_dir)
    if not (idx / 'faiss.index').exists():
        print(f"Error: no faiss.index in {idx} — run ingest-all first.")
        sys.exit(1)

    from .serving.search_api import load_search_index, run_search_pipeline
    if not load_search_index(str(idx)):
        print(f"Error: failed to load search index from {idx}")
        sys.exit(1)

    resp = run_search_pipeline(
        args.query, limit=args.top, diversity_weight=args.diversity_weight,
        max_gap_sec=args.max_gap_sec, include_captions=True)
    results = resp.results

    print(f'\nQuery: "{args.query}"')
    print(f'Top {len(results)} sequences (of {resp.total_candidates} candidate windows, '
          f'diversity={resp.diversity_score:.2f}):\n')
    if not results:
        print("  (no results)")
        return
    for i, r in enumerate(results, 1):
        dur = (r.end_time - r.start_time) if (r.end_time is not None and r.start_time is not None) else 0.0
        merged = f" merged×{r.merged_count}" if getattr(r, 'merged_count', 1) and r.merged_count > 1 else ""
        print(f"{i:2d}. [{r.score:.3f}] {r.video_id}")
        print(f"     @ {r.start_time:7.1f}s – {r.end_time:.1f}s  "
              f"(frames {r.start_frame_idx}–{r.end_frame_idx}, {dur:.1f}s{merged})")
        if r.caption:
            cap = ' '.join(r.caption.split())
            print(f"     {cap[:240]}")
        print()


def cmd_ingest_all(args):
    """
    One-command ingestion: scan + generate captions + generate embeddings.
    
    Creates a complete searchable index from raw data.
    Now uses vLLM API for caption generation.
    """
    from .processing.metadata import scan_data_directory
    from .processing.embedder import embed_all_videos_phase1_captions, embed_all_videos_phase2_embeddings
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = output_dir / 'video_metadata.json'
    
    # Step 1: Scan
    print("=" * 60)
    print("Step 1/3: Scanning data directory...")
    print("=" * 60)
    
    index = scan_data_directory(args.data_dir, pattern=args.pattern)
    index.save_json(str(metadata_path))
    
    print(f"  Found {index.count()} videos in {len(index.sessions())} sessions")
    
    if index.count() == 0:
        print("Error: No videos found. Check --data-dir and --pattern")
        sys.exit(1)
    
    # Step 2: Generate captions using vLLM
    print()
    print("=" * 60)
    print("Step 2/3: Generating captions with vLLM API...")
    print("=" * 60)
    
    stats_phase1 = embed_all_videos_phase1_captions(
        metadata_path=str(metadata_path),
        window_size_sec=args.window_size,
        batch_size=args.batch_size,
        model_name=getattr(args, 'model', 'gemma4-captioner'),
        vllm_api_base=getattr(args, 'vllm_api_base', 'http://localhost:8000/v1'),
        processed_dir=getattr(args, 'processed_dir', None),
        resume=getattr(args, 'resume', False)
    )
    
    print(f"  Captions generated: {stats_phase1['count']}")
    
    # Step 3: Generate embeddings from captions
    print()
    print("=" * 60)
    print("Step 3/3: Generating embeddings from captions...")
    print("=" * 60)
    
    stats_phase2 = embed_all_videos_phase2_embeddings(
        metadata_path=str(metadata_path),
        output_dir=str(output_dir),
        window_size_sec=args.window_size,
        batch_size=args.batch_size,
        model_name=getattr(args, 'model', 'gemma4-captioner'),
        processed_dir=getattr(args, 'processed_dir', None)
    )
    
    print()
    print("=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"  Videos: {stats_phase2['videos_processed']}")
    print(f"  Embeddings: {stats_phase2['count']}")
    print(f"  Time: {stats_phase1['elapsed_sec'] + stats_phase2['elapsed_sec']:.1f}s")
    print(f"  Index: {args.output_dir}/")
    print()
    print("Next: Start the server with:")
    print(f"  python -m sem_search.main serve --index-dir {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Video Search for Robotics Logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start the web server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    serve_parser.add_argument('--index-dir', default='./search_index', help='Directory with embeddings index')
    
    # Query command (standalone CLI search, no server)
    DEFAULT_INDEX = '/vast/projects/pratikac/multi-modal-foundation/octo/search_index'
    query_parser = subparsers.add_parser('query', help='CLI search: top-N sequences + time location for a category (no server)')
    query_parser.add_argument('query', help='Natural-language search query / category')
    query_parser.add_argument('--top', type=int, default=10, help='Number of sequences to return (default 10)')
    query_parser.add_argument('--index-dir', default=DEFAULT_INDEX, help='Search index directory')
    query_parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Query-encoder device')
    query_parser.add_argument('--diversity-weight', type=float, default=0.3, help='Diversity penalty (0-1)')
    query_parser.add_argument('--max-gap-sec', type=float, default=10.0, help='Max gap to merge adjacent windows')

    # Ingest-all command (recommended - combines scan + embed)
    ingest_all_parser = subparsers.add_parser('ingest-all', help='Scan data and create searchable index')
    ingest_all_parser.add_argument('--data-dir', required=True, help='Base data directory')
    ingest_all_parser.add_argument('--pattern', default='sess*/rosbag2_*/', help='Directory pattern (e.g., sess[789]/rosbag2_*/)')
    ingest_all_parser.add_argument('--output-dir', default='./search_index', help='Output directory for index')
    ingest_all_parser.add_argument('--window-size', type=float, default=5.0, help='Window size in seconds')
    ingest_all_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for GPU')
    ingest_all_parser.add_argument('--model', default='gemma4-captioner',
                                   help='Model to use. Default: gemma4-captioner (Gemma4-31B).')
    ingest_all_parser.add_argument('--vllm-api-base', default='http://localhost:8000/v1',
                                   help='vLLM API base URL for caption generation')
    ingest_all_parser.add_argument('--processed-dir', default=None,
                                   help='Base directory for per-sequence H5 files (e.g., /data/rosbags/processed/car). If not specified, inferred from video paths.')
    ingest_all_parser.add_argument('--resume', action='store_true',
                                   help='Skip videos that already have captions (resume from failure)')
    
    
    args = parser.parse_args()
    
    if args.command == 'serve':
        cmd_serve(args)
    elif args.command == 'query':
        cmd_query(args)
    elif args.command == 'ingest-all':
        cmd_ingest_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
