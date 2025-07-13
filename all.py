import subprocess

def run_script(script_name):
    print(f"\n📂 Running: {script_name}")
    result = subprocess.run(["python", script_name], check=True)
    print(f"✅ Finished: {script_name}")

if __name__ == "__main__":
    try:
                       # For capturing or creating dataset
        run_script("prepare_dataset.py")    # For preparing X, y arrays
        run_script("train_model.py")        # For training and saving the model
        run_script("sent.py")    # For live gesture prediction (opens UI)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running: {e.cmd}")
        print("🔁 Stopping further execution due to an error.")
