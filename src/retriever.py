import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunking_baseline(path):
    md = pymupdf4llm.to_markdown(path)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sections = text_splitter.split_text(md)

    return sections

def chunking_improved(md, path):
    paper_title = path.split("\\")[-1]
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sections = header_splitter.split_text(md)

    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []
    metadatas = []

    for section in sections:
        section_text = section.page_content
        section_meta = section.metadata

        split_chunks = chunk_splitter.split_text(section_text)

        for i, chunk in enumerate(split_chunks):

            metadata = {
                "paper": paper_title,
                "section": section_meta.get("Header 1", "Unknown"),
                "subsection": section_meta.get("Header 2", ""),
                "chunk_id": len(chunks)
            }

            chunks.append(chunk)
            metadatas.append(metadata)

    return chunks, metadatas

def chunking_multiple(paths):

    all_chunks = []
    all_metadatas = []

    for path in paths:
        md = pymupdf4llm.to_markdown(path)
        chunks, metadatas = chunking_improved(md, path)
        all_chunks.extend(chunks)
        all_metadatas.extend(metadatas)

    for i, metadata in enumerate(all_metadatas):
        metadata["chunk_id"] = i

    return all_chunks, all_metadatas
