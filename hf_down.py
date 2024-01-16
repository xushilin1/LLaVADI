import argparse

from huggingface_hub import snapshot_download


def main(args):

    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        ignore_patterns=[".gitattributes", "README.md", "*.jpg", "*.fp16.*", "*non_ema*"],
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        max_workers=8,
        resume_downloan=True,
        cache_dir='./cache_dir/'
)


if __name__ == '__main__':
    """
    HF_ENDPOINT=https://hf-mirror.com python hf_down.py
    """
    import os
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument('local_dir')
    parser.add_argument('--repo_type', default='model')
    args = parser.parse_args()
    main(args)