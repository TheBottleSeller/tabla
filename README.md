# Tabla

## Setup
```
cd tools/utilFunctions_C
python compileModule.py build_ext --inplace
```

### Validating Audio Files:
```
python validate_audio_files.py --audio_file_path ~/Google\ Drive/Tabla/Pneumonia\ Study\ Data/ --study PNA
```

### Running Eko Scraper
- Go to: https://github.com/TheBottleSeller/tabla/blob/master/scraping.js
- Copy contents of file
- Login to Eko and go to patient profile (the patient data you want to download)
- Open Developer Console by hitting `Option + Command + i`
- Go to the "Console" tab
- Paste the contents of the scraping file into the console and hit [ENTER]
- Type in `scrapeAudio("<patient id>")` and hit [ENTER]
- Wait for "Successs! ..." and copy output
- Open terminal and paste output into terminal and hit [ENTER]
