# src/app.py
import gradio as gr
from src.orchestrator import run_automl_pipeline
from dotenv import load_dotenv
load_dotenv()

def automl_interface(csv_file, problem_text):
    if not csv_file or not problem_text.strip():
        return "Please upload a CSV and enter a problem description.", ""

    try:
        # Run the pipeline and get the output
        output = run_automl_pipeline(csv_file.name, problem_text.strip())

        # Extract logs and summary
        log_text = "\n".join(output["logs"])

        summary = f"""
**FINAL RESULT**

**Best Model**: {output['final_evaluation']['best_model']}  
**Score**: {output['final_evaluation']['best_score']:.4f} ({output['final_plan'].primary_metric})  
**Solved?**: {'YES' if output['final_evaluation']['solved'] else 'NO'}  
**Iterations**: {output['iterations']}

**Critique**:  
{output['final_evaluation']['critique']}

**Suggestion** (if not solved):  
{output['final_evaluation']['suggestion'] or 'None'}
        """.strip()

        return summary, log_text

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"**Error**: {str(e)}\n\nDetails:\n{error_detail}", ""

# Gradio interface
with gr.Blocks(title="Structured LLM-Agent AutoML") as demo:
    gr.Markdown("# Structured LLM-Agent AutoML System")
    gr.Markdown("Upload a tabular dataset and describe the prediction task. The agents will autonomously plan, execute, and improve until solved.")

    with gr.Row():
        csv_input = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
        problem_input = gr.Textbox(
            label="Problem Description",
            placeholder="e.g., Predict whether a client will subscribe to a term deposit",
            lines=3
        )

    run_btn = gr.Button("Run AutoML Pipeline", variant="primary")

    with gr.Row():
        summary_output = gr.Markdown(label="Summary")
        log_output = gr.Textbox(label="Full Logs", lines=20)

    run_btn.click(
        fn=automl_interface,
        inputs=[csv_input, problem_input],
        outputs=[summary_output, log_output]
    )

    gr.Markdown("### Example Datasets in `data/` folder: iris.csv, wine.csv, adult.csv, bank.csv")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)