import gradio as gr
import final_project as fp
def gradio_wrapper(ticket_text):
    """
    Wrapper function to format the output for Gradio.
    """
    # 1. Get results from the main pipeline function
    result = fp.process_ticket(ticket_text)
    
    # 2. Return the specific components
    # We return them in the order corresponding to the 'outputs' list below
    return (
        result['predicted_issue_type'], 
        result['predicted_urgency_level'], 
        result['extracted_entities']
    )

# -----------------------------------------------------------
# Define the Gradio Interface
# -----------------------------------------------------------
app = gr.Interface(
    fn=gradio_wrapper,
    inputs=gr.Textbox(
        lines=5, 
        placeholder="e.g., My internet is down and I need a refund...", 
        label="Customer Ticket Text"
    ),
    outputs=[
        gr.Label(label="Predicted Issue Type"),
        gr.Label(label="Predicted Urgency Level"),
        gr.JSON(label="Extracted Entities")
    ],
    title="Support Ticket AI Classifier",
    description="Enter a customer support message to automatically classify the issue, assess urgency, and extract key details (products, dates, etc.).",
    theme="default",
    examples=[
        ["My laptop screen is flickering constantly!!"],
        ["I need a refund for order #998877. It arrived broken."],
        ["URGENT: Server crash in the main database."]
    ]
)

# -----------------------------------------------------------
# Launch the App
# -----------------------------------------------------------
if __name__ == "__main__":
    # set share=True to create a public link (accessible from other devices)
    app.launch(share=False)