# Import standard os library to handle temporary files
import os
# Import gradio for building interface UI
import gradio as gr
# Import predict function from prediction script
from predict import predict_star_class, CLASS_NAMES

# Define the prediction wrapper function for Gradio interface
def gradio_predict(file_obj):
    # If no file uploaded, return empty predictions
    if file_obj is None:
        return "Please upload a valid FITS or CSV file.", None, None

    # Get local path of the uploaded file
    file_path = file_obj.name
    
    # Run prediction pipeline
    try:
        pred_label, confidence = predict_star_class(file_path)
    except Exception as e:
        return f"Error processing light curve: {e}", None, None
        
    # The predict_star_class function generates 'lightcurve_diagnostic_plot.png'
    diagnostic_plot_path = "lightcurve_diagnostic_plot.png"
    
    if not os.path.exists(diagnostic_plot_path):
        diagnostic_plot_path = None
        
    # Create HTML response summary of classification result
    result_html = f"""
    <div style="text-align: center; font-family: sans-serif; padding: 10px;">
        <h2 style="color: #4A90E2; margin-bottom: 5px;">Prediction: {pred_label}</h2>
        <h3 style="color: #555; margin-top: 0;">Confidence Score: {confidence:.2f}%</h3>
    </div>
    """
    
    return result_html, diagnostic_plot_path

# Create custom theme and interface layout using Gradio Blocks
with gr.Blocks(title="Andromeda Stars Classifier", theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # 🌌 Andromeda Stars Classifier
        Upload a star's light curve file (.fits or .csv) to analyze its periodicity using **Lomb-Scargle Phase Folding** and predict its variable star class via a **1D Convolutional Neural Network (CNN)**.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # File upload component
            file_input = gr.File(
                label="Upload Light Curve (.fits / .csv)", 
                file_types=[".fits", ".csv"], 
                type="filepath"
            )
            # Submit button
            submit_btn = gr.Button("Analyze and Classify", variant="primary")
            
        with gr.Column(scale=2):
            # Output fields
            output_html = gr.HTML(label="Results Summary")
            output_image = gr.Image(label="Lomb-Scargle & Light Curve Analysis Plots", type="filepath")

    # Link submit button click to calculation wrapper function
    submit_btn.click(
        fn=gradio_predict,
        inputs=file_input,
        outputs=[output_html, output_image]
    )
    
    # Add examples section to help the user test out the UI
    gr.Examples(
        examples=[],
        inputs=file_input,
        label="Quick Test Examples (Generate one using predict.py to test!)"
    )

# Run the app locally if executed directly
if __name__ == "__main__":
    demo.launch(server_port=7860)
