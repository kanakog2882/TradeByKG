# 🚀 SOP v7.4 - Professional Options Trading Dashboard

Real-time intraday options assistant with evolving SOP logic for NIFTY options trading.

---

## How to Run

1. **Install requirements**

    ```
    pip install -r requirements.txt
    ```

2. **Configure your Dhan API credentials**

    - Create or edit the `.streamlit/secrets.toml` file in your project directory.
    - Paste this content, replacing with your actual credentials:

      ```
      [dhan]
      client_id = "YOUR_CLIENT_ID"
      access_token = "YOUR_TOKEN_KEY"
      ```

    - **Do NOT share this file or commit your real credentials to a public repository.**

3. **Run the app (use the correct filename):**

    ```
    streamlit run streamlit_app.py
    ```

    *(If your app's main file uses a different name, replace `streamlit_app.py` accordingly.)*

4. **Open the dashboard**

    - After launching, use the provided local URL (from the terminal output) or access the forwarded URL in GitHub Codespaces.

---

## Troubleshooting

- **Stuck on "🟡 DEMO DATA"?**
  - Your Dhan API credentials may be missing, invalid, or not detected.
  - Double-check the section names (`[dhan]`) and variable names (`client_id`, `access_token`) in `.streamlit/secrets.toml`.
  - Restart the app after editing secrets.

- **Logs and Errors:**  
  - If you encounter errors, check the terminal/Streamlit server output for details.

---

## Development & Contributions

- Requirements: List all dependencies in `requirements.txt`.
- To contribute: Open issues or pull requests. Forks welcome!

---

## Security

- **Keep your `.streamlit/secrets.toml` private!**  
- Never push real API credentials to any public repo.

---
