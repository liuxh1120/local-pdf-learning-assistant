from pdf_assistant.ui import APP_CSS, build_ui

if __name__ == "__main__":
    demo, settings = build_ui()
    demo.queue(default_concurrency_limit=2).launch(
        server_name=settings.app_host,
        server_port=settings.app_port,
        inbrowser=True,
        show_error=True,
        css=APP_CSS,
    )
