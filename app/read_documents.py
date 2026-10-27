from pathlib import Path

DOCUMENT_DIR = Path("/app/data/sample")


def split_sections(text):
    chunks = []
    current_section = "Introduction"
    current_lines = []

    for line in text.splitlines():

        # Start a new chunk at each Markdown heading.
        if line.startswith("## "):

            content = "\n".join(current_lines).strip()

            if content:
                chunks.append({
                    "section": current_section,
                    "text": content
                })

            current_section = line[3:].strip()
            current_lines = []

        elif not line.startswith("# "):
            current_lines.append(line)

    # Save the final section.
    content = "\n".join(current_lines).strip()

    if content:
        chunks.append({
            "section": current_section,
            "text": content
        })

    return chunks


def main():

    # Find Markdown documents in our directory.
    files = sorted(DOCUMENT_DIR.glob("*.md"))

    for file in files:

        print(f"\nReading: {file.name}")

        # Read the document as text.
        text = file.read_text(encoding="utf-8")

        # Split the document into sections.
        chunks = split_sections(text)

        for index, chunk in enumerate(chunks):

            print(f"\nChunk {index + 1}")
            print(f"Source: {file.name}")
            print(f"Section: {chunk['section']}")
            print(f"Text: {chunk['text']}")
            print("-" * 40)


if __name__ == "__main__":
    main()
