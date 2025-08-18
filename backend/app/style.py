MAIN_STYLE="""
        <style>
          @font-face { font-family: 'Quicksand Variable'; font-style: normal; font-weight: 300 700; src: url('https://arnodell.hopto.org/fonts/Quicksand-VariableFont_wght.ttf') format('truetype-variations'); font-display: swap; }
          @font-face { font-family: 'Quicksand'; font-style: normal; font-weight: 300; src: url('https://arnodell.hopto.org/fonts/Quicksand-Light.ttf') format('truetype'); font-display: swap; }
          @font-face { font-family: 'Quicksand'; font-style: normal; font-weight: 400; src: url('https://arnodell.hopto.org/fonts/Quicksand-Regular.ttf') format('truetype'); font-display: swap; }
          @font-face { font-family: 'Quicksand'; font-style: normal; font-weight: 500; src: url('https://arnodell.hopto.org/fonts/Quicksand-Medium.ttf') format('truetype'); font-display: swap; }
          @font-face { font-family: 'Quicksand'; font-style: normal; font-weight: 600; src: url('https://arnodell.hopto.org/fonts/Quicksand-SemiBold.ttf') format('truetype'); font-display: swap; }
          @font-face { font-family: 'Quicksand'; font-style: normal; font-weight: 700; src: url('https://arnodell.hopto.org/fonts/Quicksand-Bold.ttf') format('truetype'); font-display: swap; }
          body, input, button, select, h1, h2, h3, h4, h5, h6, p, msg   { font-family: 'Quicksand', 'Quicksand Variable', sans-serif; }
                          html {
                        height: 100%;
                        -webkit-background-size: cover;
                        -moz-background-size: cover;
                        -o-background-size: cover;
                        background-size: cover;
                        background-repeat: no-repeat;
                        background: -moz-linear-gradient(20deg, rgba(95, 60, 240, 1) 0%, rgba(6, 11, 54, 51) 100%);
                        background: -webkit-linear-gradient(20deg, rgba(95, 60, 240, 1) 0%, rgba(6, 11, 54, 51) 100%);
                        background: linear-gradient(20deg, rgba(95, 60, 240, 1) 0%, rgba(6, 11, 54, 51) 100%);
                        color: #ffffff;
                        text-align: center;
                        font-weight: 100;
                        font-size: 16px;
                }

          body { margin: 0; padding: 2rem; background: transparent; color: rgba(230,230,255,0.8); }
          .card { max-width: 640px; margin: 0 auto; background: rgba(0,0,0, 0.3); border-radius: 9px; padding: 12px; box-shadow: 0 6px 24px rgba(0,0,0,0.25); }
          h1 { font-weight: 600; font-size: 1.25rem; margin: 0 0 1rem; }
          .row { display: flex; gap: 12px; align-items: center; }
          select { flex: 1; padding: 10px; border: 1px solid; color: rgba(230,230,255,0.8); border-radius: 8px; background: rgba(255,255,255,0.1); }
          select option { background-color: rgba(30,10,80,0.9); color: rgba(200,200,255,0.8); }
          button { padding: 10px 16px; border: 0; background: rgba(255,255,255,0.2); color: rgba(230,230,255,0.8); border-radius: 8px; cursor: pointer; font-weight: 600; }
          button:disabled { background: #9aa7cc; cursor: not-allowed; }
          .msg { margin-top: 12px; font-size: 0.95rem; }
          .ok { color: #0a7d32; }
          .err { color: #b00020; white-space: pre-wrap; }
        </style>"""
