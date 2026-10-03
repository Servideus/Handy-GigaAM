[English documentation](README.md).

# Handy-GigaAM

Экспериментальная производная [Handy](https://github.com/cjpais/Handy) с интеграцией GigaAM и изменениями сборки Windows/Vulkan. Handy разработан CJ Pais и участниками исходного проекта. Этот репозиторий не является приложением, написанным с нуля. GigaAM уже поддерживается в upstream; здесь сохранены эксперимент и дополнительные исправления сборки.

## Статус и собственный вклад

[Опубликованный коммит](https://github.com/Servideus/Handy-GigaAM/commit/922cfceff18f9f42026afbb0c47cfdfb671aa731) включает Vulkan для Whisper, Ninja для вложенных CMake-сборок, короткий каталог Cargo target и изменения GigaAM/UI.

Сборка интерфейса, lint и `cargo fmt --check` прошли. Полная Tauri-сборка и clippy остановились из-за отсутствующего MSVC `link.exe`. Дополнительные локальные эксперименты с моделями и пунктуацией не опубликованы. См. [подробности вклада и проверки](docs/GIGAAM_CONTRIBUTIONS.md).

## Как работает Handy

Нажмите настроенное сочетание клавиш, произнесите текст и завершите запись. Приложение отфильтрует тишину с Silero VAD, распознает речь локальной моделью и вставит текст в активное поле через буфер обмена. Можно использовать переключение записи или удержание клавиши.

В исходном Handy доступны Whisper и Parakeet; набор моделей этой производной смотрите в опубликованном коде и настройках. Поддержка ускорения зависит от платформы и оборудования. Локальное распознавание не требует отправки аудио в облако.

## Установка и запуск

Готовые установщики на [странице релизов Handy](https://github.com/cjpais/Handy/releases) и [handy.computer](https://handy.computer) относятся к исходному проекту, а не к проверенной сборке этой производной.

После установки разрешите доступ к микрофону и необходимые системные разрешения, выберите модель и настройте сочетание клавиш. На macOS для работы клавиатурных действий нужны разрешения универсального доступа.

Для сборки этой версии используйте [BUILD.md](BUILD.md). Нужны stable Rust и Bun; Windows-конфигурация также требует MSVC C++ Build Tools, WebView2, Ninja и Vulkan SDK. Сборку на Windows запускайте из x64 Developer Terminal.

```sh
bun install
bun run tauri dev
bun run tauri build
```

Только интерфейс: `bun run dev`, `bun run build`, `bun run preview`. Это не проверка записи аудио, распознавания или вставки текста. Файл VAD и платформенные зависимости описаны в `BUILD.md`.

## Управление и диагностика

| Флаг | Действие |
|---|---|
| `--toggle-transcription` | Начать или остановить запись работающего экземпляра |
| `--toggle-post-process` | Переключить запись с последующей обработкой |
| `--cancel` | Отменить текущую операцию |
| `--start-hidden` | Запуститься без окна, с иконкой в трее |
| `--no-tray` | Запуститься без трея; закрытие окна завершает процесс |
| `--debug` | Подробный журнал |

Флаги не меняют сохранённые настройки. Второй экземпляр передаёт команды уже работающему. Меню диагностики: `Cmd+Shift+D` на macOS, `Ctrl+Shift+D` на Windows/Linux. Интеграция [Raycast](https://www.raycast.com/mattiacolombomc/handy) относится к Handy.

## Модели при проблемах с сетью

В разделе About скопируйте App Data Directory и создайте внутри каталог `models`. Обычные расположения:

| Система | Каталог |
|---|---|
| Windows | `%APPDATA%\com.pais.handy\models` |
| macOS | `~/Library/Application Support/com.pais.handy/models` |
| Linux | `~/.config/com.pais.handy/models` |

Whisper GGML-файлы `.bin` помещаются непосредственно в `models`; имена скачанных файлов нужно сохранить. Архивы Parakeet распаковываются в `parakeet-tdt-0.6b-v2-int8` или `parakeet-tdt-0.6b-v3-int8`. После перезапуска выберите модель в настройках и проверьте диктовку. Пользовательские GGML `.bin` обнаруживаются в этом же каталоге.

Ссылки на файлы и точная структура приведены в [английском руководстве](README.md#manual-model-installation-for-proxy-users-or-network-restrictions). Эта инструкция для Whisper/Parakeet не заменяет формат GigaAM: сверяйте его с каталогом моделей в коде.

## Linux и известные ограничения

Whisper может аварийно завершаться на некоторых Windows/Linux-конфигурациях. Wayland поддерживается ограниченно. Для ввода текста в X11 нужен `xdotool`, в Wayland — `wtype` или `dotool`; для `dotool` требуется группа `input` и повторный вход в систему.

В Wayland глобальные сочетания задавайте средствами GNOME/KDE или конфигурацией оконного менеджера. Команда для сочетания: `handy --toggle-transcription`. Unix-сигналы `SIGUSR2` и `SIGUSR1` переключают обычное распознавание и распознавание с обработкой соответственно.

Если окно записи забирает фокус, в Settings → Advanced установите Overlay Position → None; для подтверждения записи можно включить Audio Feedback.

При ошибке `libgtk-layer-shell.so.0` установите runtime-пакет `libgtk-layer-shell0` для Ubuntu/Debian или `gtk-layer-shell` для Fedora/Arch. Если библиотека установлена, но запуск нестабилен, попробуйте по одному обходному варианту:

```sh
HANDY_NO_GTK_LAYER_SHELL=1 handy
WEBKIT_DISABLE_DMABUF_RENDERER=1 handy
```

Сначала проверьте каждый вариант отдельно; в автозапуск добавляйте только тот, который помог. Подробные платформенные замечания сохранены в [руководстве Handy](README.md#linux-notes). Рекомендации upstream для Whisper включают GPU на Windows/Linux; Parakeet рассчитан на CPU. Производительность этой производной отдельно не измерена.

## Проверка подписей и участие

Upstream-релизы используют подписи обновления Tauri. Ключ находится в `src-tauri/tauri.conf.json`, поле `plugins.updater.pubkey`. Для ручной проверки декодируйте ключ и `.sig` из base64 и используйте `minisign`, а не `gpg`: [полная команда](README.md#verify-release-signatures). Это не подтверждает подпись иной сборки или авторство данного форка.

Перед изменениями выполните lint и форматирование по [AGENTS.md](AGENTS.md). Для участия в исходном Handy прочитайте [CONTRIBUTING.md](CONTRIBUTING.md), [правила переводов](CONTRIBUTING_TRANSLATIONS.md) и шаблоны GitHub. Актуальное развитие, ошибки и спонсоры перечислены в исходном README; ссылки на релизы ведут в upstream.

## Технологии и лицензия

Tauri, React/TypeScript, Rust, Whisper/Parakeet, GigaAM, Silero VAD, cpal и rubato. Исходная [MIT-лицензия](LICENSE) и авторство Handy сохранены. Модели и зависимости имеют собственных авторов и условия. Связанные проекты: [Handy CLI](https://github.com/cjpais/handy-cli) и [handy.computer](https://handy.computer).
