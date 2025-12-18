import gradio as gr
import subprocess
import os
import shutil
import glob
import time

def run_inference(image, caption, edit_prompt):
    # Create a unique output directory for this run
    timestamp = int(time.time())
    run_dir = f"gradio_output/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save input image
    input_image_path = os.path.join(run_dir, "input.png")
    image.save(input_image_path)
    
    # Construct the command
    # Using the parameters from the user's last successful run
    cmd = [
        "torchrun", "--nnodes=1", "--nproc_per_node=1", "train.py",
        "--image_file_path", input_image_path,
        "--image_caption", caption,
        "--editing_prompt", edit_prompt,
        "--output_dir", run_dir,
        "--diffusion_model_path", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "--test_alpha", "2",
        "--test_beta", "1",
        # "--use_dashscope_caption",
        # "--use_dashscope_edit_description",
        "--dashscope_api_key", "sk-5463a1be1a5748c9ad825a54b2bceeec",
        "--draw_box",
        "--lr", "5e-3",
        "--max_window_size", "15",
        "--per_image_iteration", "1",
        "--epochs", "1",
        "--num_workers", "8",
        "--seed", "42",
        "--pin_mem",
        "--point_number", "9",
        "--batch_size", "1",
        "--save_path", os.path.join(run_dir, "checkpoints")
    ]
    
    # Set environment variables to suppress warnings
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Error running script:")
        print(e.stdout)
        print(e.stderr)
        return [], None, f"Error: {e.stderr}"

    # Find the results
    # The script creates a subdirectory inside run_dir named "0_{caption}"
    # We need to find it.
    subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and d != "checkpoints"]
    if not subdirs:
        return [], None, "Error: No output directory found."
    
    # Assuming there's only one relevant subdirectory
    result_dir = os.path.join(run_dir, subdirs[0], "results")
    box_dir = os.path.join(run_dir, subdirs[0], "boxes")
    
    if not os.path.exists(result_dir):
        return [], [], None, f"Error: Result directory {result_dir} not found."
    
    # Collect images
    # We look for images that are NOT the final output first
    all_images = glob.glob(os.path.join(result_dir, "*.png"))
    
    final_image_path = os.path.join(result_dir, "final_output.png")
    
    candidate_images = []
    final_image = None
    
    for img_path in all_images:
        if os.path.basename(img_path) == "final_output.png":
            final_image = img_path
        elif "anchor" in os.path.basename(img_path) and "ori_draw" not in os.path.basename(img_path):
             # Filter for the generated candidates (usually named with 'anchor')
             # Excluding 'ori_draw' which are likely debug images with boxes drawn on original
             candidate_images.append(img_path)
    
    # Sort candidates to ensure consistent order if needed
    candidate_images.sort()

    # Collect box images
    box_images = []
    if os.path.exists(box_dir):
        all_box_images = glob.glob(os.path.join(box_dir, "*.png"))
        for img_path in all_box_images:
             # Usually these are named with 'ori_draw'
             box_images.append(img_path)
        box_images.sort()
    
    return candidate_images, box_images, final_image

# Define Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Learnable Regions Demo")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image", height=500)
            caption = gr.Textbox(label="Original Caption", value="trees")
            edit_prompt = gr.Textbox(label="Editing Prompt", value="a big tree with many flowers in the center")
            run_btn = gr.Button("Run")
            # status = gr.Textbox(label="Status")
        
        with gr.Column(scale=1):
            final_output = gr.Image(label="Final Selected Image", height=300)
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Box Visualizations")
            box_gallery = gr.Gallery(label="Box Visualizations", columns=3, rows=3, height=500, object_fit="contain")
        with gr.Column():
            gr.Markdown("### Candidate Images")
            gallery = gr.Gallery(label="Candidate Images", columns=3, rows=3, height=500, object_fit="contain")


    run_btn.click(
        fn=run_inference,
        inputs=[input_image, caption, edit_prompt],
        outputs=[gallery, box_gallery, final_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
