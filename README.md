# Очистка голоса

Локальный сервис: на вход урок в m4a, на выход mp3 той же длины. Остаётся голос. Вздохи, кашель, щелчки и удары заглушаются на месте, без склейки и без облачных API.

Второй раздел принимает видео целиком (mp4, mov, mkv, webm). Голос очищается тем же способом, картинка пересобирается в mp4 не тяжелее 48 МБ, чтобы ролик было удобно отправить в Telegram.

Страница открывается только на этом компьютере: [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## Что нужно

- Python 3.12
- ffmpeg (`ffmpeg` и `ffprobe` в `PATH`)
- на Mac для сборки DeepFilterNet ещё Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y`, затем `source "$HOME/.cargo/env"`

На Mac ffmpeg ставится так: `brew install ffmpeg`.

## Запуск

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app
```

Первый запуск скачает модели (DeepFilterNet, Silero VAD, YAMNet) и может занять несколько минут. Дальше сервис поднимается командой `python -m app` из этой папки. Окно терминала не закрывать. Остановка: Control+C.

Проверка без нейросетей: `python -m app.selftest`.

Обработка файла из терминала:

```bash
python -m app.pipeline "урок.m4a" "voice.mp3"
```
