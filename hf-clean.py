from huggingface_hub import scan_cache_dir

def clean_hf_cache(keep_model="gemma-4-E2B"):
    cache_info = scan_cache_dir()
    repos_to_delete = []
    
    for repo in cache_info.repos:
        # Check if this repo is NOT the one we want to keep
        if keep_model.lower() not in repo.repo_id.lower():
            repos_to_delete.append(repo)
            print(f"Marked for deletion: {repo.repo_id} ({repo.size_on_disk_str})")

    if not repos_to_delete:
        print("No unwanted models found in cache.")
        return

    # Total size to be freed
    total_freed = sum(repo.size_on_disk for repo in repos_to_delete)
    confirm = input(f"\nDelete these {len(repos_to_delete)} models? (Freed space: {total_freed / 1e9:.2f} GB) [y/N]: ")

    if confirm.lower() == 'y':
        delete_strategy = cache_info.delete_revisions(*[
            revision.commit_hash 
            for repo in repos_to_delete 
            for revision in repo.revisions
        ])
        delete_strategy.execute()
        print("Cleanup complete.")
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    clean_hf_cache()