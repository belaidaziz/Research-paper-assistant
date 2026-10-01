import gradio as gr
from main import query_pipeline_multi


def answer_from_pdfs(pdf_files, question):
    if not pdf_files:
        return "Please upload at least one PDF."
    pdf_paths = [f.name for f in pdf_files]

    answer = query_pipeline_multi(question, pdf_paths, use_cache=False)

    paper_list = "\n".join(f"- {p.split('/')[-1]}" for p in pdf_paths)
    return f"Papers used:\n{paper_list}\n\nQuestion: {question}\n\nAnswer:\n{answer}"


demo = gr.Interface(
    fn=answer_from_pdfs,
    inputs=[
        gr.File(label="Upload your PDF(s)", file_types=[".pdf"], file_count="multiple"),
        gr.Textbox(label="Ask a question", lines=2, placeholder="e.g., What is the summary?")
    ],
    outputs=gr.Textbox(label="Answer", lines=10),
    title="Research paper assistant",
    description="Upload one or more PDFs and ask a question across all of them."
)

demo.launch()