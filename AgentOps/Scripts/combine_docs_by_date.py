from pathlib import Path
from datetime import datetime

def combine_markdown_by_date():
    """
    Recursively traverses a target directory for markdown files,
    combines them into a single file, ordered chronologically by creation date.
    """
    target_dir = Path("docs")
    if not target_dir.is_dir():
        print(f"Error: The directory '{target_dir}' was not found in the current workspace.")
        return

    output_filename = "_COMBINED_BY_DATE.md"
    output_path = target_dir / output_filename

    print(f"Starting to process markdown files in '{target_dir}'...")

    # Find all markdown files RECURSIVELY.
    # The check against output_path prevents the combined file from being included.
    try:
        all_markdown_files = [
            p for p in target_dir.glob("**/*.md") if p.resolve() != output_path.resolve()
        ]
    except Exception as e:
        print(f"Error during file discovery: {e}")
        return

    if not all_markdown_files:
        print(f"  -> No markdown files found in {target_dir} to process. Skipping.")
        return

    # Sort files by creation time (st_ctime).
    # Note: On Unix, ctime is the last metadata change time. On Windows, it's creation time.
    try:
        sorted_files = sorted(all_markdown_files, key=lambda p: p.stat().st_ctime)
    except FileNotFoundError as e:
        print(f"Error accessing file for sorting: {e}. A file may have been deleted during the run.")
        return

    print(f"\nFound {len(sorted_files)} markdown files. Combining in chronological order...")

    combined_content = []

    for md_file in sorted_files:
        try:
            stat_info = md_file.stat()
            # Using fromtimestamp to convert ctime (float) to a datetime object
            creation_time = datetime.fromtimestamp(stat_info.st_ctime)
            formatted_time = creation_time.strftime('%Y-%m-%d %H:%M:%S')

            print(f"  -> Reading: {md_file} (Created: {formatted_time})")

            # Use .as_posix() to ensure forward slashes for display in the header.
            relative_path = md_file.as_posix()

            # Create the spacer with the relative file path and creation date
            spacer = (
                "\n\n---\n---\n---\n\n"
                f"# <<< File: {relative_path} | Created: {formatted_time} >>>\n\n"
                "---\n---\n---\n\n"
            )

            combined_content.append(spacer)

            # Read the content of the markdown file
            content = md_file.read_text(encoding="utf-8")
            combined_content.append(content)

        except FileNotFoundError:
            print(f"    -> Warning: File not found, it may have been deleted during script execution: {md_file}")
        except Exception as e:
            print(f"    -> Error processing file {md_file}: {e}")

    # Write the combined content to the new file
    if combined_content:
        print(f"\n  -> Writing combined file to: {output_path}")
        try:
            output_path.write_text("".join(combined_content), encoding="utf-8")
        except Exception as e:
            print(f"    -> Error writing file {output_path}: {e}")

    print("\nProcessing complete.")

if __name__ == "__main__":
    combine_markdown_by_date()