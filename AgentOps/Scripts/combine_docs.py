import os
from pathlib import Path

def combine_markdown_files():
    """
    Iterates through subdirectories of a 'docs' folder, combines all .md files
    within each subdirectory into a single file, separated by a clear spacer.
    """
    root_dir = Path("docs")
    if not root_dir.is_dir():
        print(f"Error: The directory '{root_dir}' was not found in the current workspace.")
        return

    print(f"Starting to process subdirectories in '{root_dir}'...")

    # Iterate through each item in the root 'docs' directory
    for subdir in root_dir.iterdir():
        if subdir.is_dir():
            print(f"\nProcessing directory: {subdir}")
            
            # Create a filename-safe prefix from the subdirectory's root-relative path
            # e.g., 'docs/00_Background' -> 'docs-00_Background'
            path_prefix = os.path.join(*subdir.parts).replace(os.sep, '-')
            
            # Construct the new output filename and place it in the root 'docs' directory
            output_filename = f"{path_prefix}-_COMBINED.md"
            output_path = root_dir / output_filename

            # Find all markdown files RECURSIVELY. The check against output_path
            # is kept for robustness, though the output file is now in a parent directory.
            markdown_files = sorted(
                p for p in subdir.glob("**/*.md") if p.resolve() != output_path.resolve()
            )
            
            if not markdown_files:
                print(f"  -> No markdown files found in {subdir} to process. Skipping.")
                continue

            combined_content = []
            
            for md_file in markdown_files:
                print(f"  -> Reading: {md_file}")
                
                # Use os.path.join to create platform-agnostic relative paths for the header
                relative_path = os.path.join(*md_file.parts)

                # Create the spacer with the relative file path
                spacer = (
                    "\n\n---\n---\n---\n\n"
                    f"# <<< File: {relative_path.replace(os.sep, '/')} >>>\n\n"
                    "---\n---\n---\n\n"
                )
                
                combined_content.append(spacer)
                
                # Read the content of the markdown file
                try:
                    content = md_file.read_text(encoding="utf-8")
                    combined_content.append(content)
                except Exception as e:
                    print(f"    -> Error reading file {md_file}: {e}")

            # Write the combined content to the new file
            if combined_content:
                print(f"  -> Writing combined file to: {output_path}")
                try:
                    output_path.write_text("".join(combined_content), encoding="utf-8")
                except Exception as e:
                    print(f"    -> Error writing file {output_path}: {e}")

    print("\nProcessing complete.")

if __name__ == "__main__":
    combine_markdown_files()
