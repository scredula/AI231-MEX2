# Validation data

Two layouts are auto-detected (manifest wins).

## 1. Bundled holdout: `test_data/holdout/` (manifest-driven)
This is the collated holdout set. Do **not** rename its columns.

```
test_data/holdout/
├── manifest.csv        # one row per clip (196 rows, 186 in-scope + 10 out-of-scope)
└── audio/              # the WAVs referenced by the manifest
```

Relevant `manifest.csv` columns:
| column | meaning |
|---|---|
| `file` | WAV path relative to `holdout/` (e.g. `audio/holdout_000123_...wav`) |
| `command` | base intent (`PLAY_MUSIC`, `WEATHER`, … for slot intents: `TIMER`, `ALARM`, `COLOR`, `BRIGHTNESS`, `TEMPERATURE`, `CREATE_REMINDER`) |
| `slot_value` | slot value, e.g. `10 seconds`, `6:00 AM`, `100 percent`, `Blue`, `Drink water`, `18 degrees` |
| `out_of_scope` | `1` = negative (non-command) → used for the **false accept rate**; `0` = in-scope command |

The notebooks map `(command, slot_value)` to the model's 31 classes
(e.g. `("TIMER","10 seconds") → TIMER_10s`, `("ALARM","6:00 AM") → ALARM_6_00AM`)
and treat `out_of_scope == 1` rows as the negatives. Extra columns
(`variation`, `bucket`, `speaker_id`, `source`, …) are ignored.

## 2. Folder-per-class (your own clips)
Add clips here if you collect your own (in addition to / instead of the holdout):


```
rpi_validation/test_data/
├── PLAY_MUSIC/        *.wav      # positives: one folder per intent you have clips for
├── WEATHER/           *.wav
├── LIGHT_ON/          *.wav
├── _negative/         *.wav      # negatives (non-command / background) -> used for FAR
└── manifest.csv                  # OPTIONAL alternative: columns  path,label
```

Rules:
- **WAV files only** (`*.wav`). Any sample rate / mono-or-stereo is fine — they are
  resampled to 16 kHz mono automatically.
- The folder name must **exactly match one of the 31 model intents** (case-sensitive):
  `ALARM_6_00AM`, `ALARM_8_00AM`, `ALARM_9_00PM`, `BRIGHTNESS_100`, `BRIGHTNESS_20`,
  `BRIGHTNESS_60`, `CALL`, `COLOR_BLUE`, `COLOR_GREEN`, `COLOR_RED`,
  `CREATE_REMINDER_DRINK_WATER`, `CREATE_REMINDER_EXERCISE`, `CREATE_REMINDER_STUDY`,
  `LIGHT_OFF`, `LIGHT_ON`, `LIST_REMINDERS`, `MESSAGE`, `NEXT`, `PAUSE`, `PLAY_MUSIC`,
  `STOP`, `TEMPERATURE_18`, `TEMPERATURE_22`, `TEMPERATURE_26`, `TIME`, `TIMER_10s`,
  `TIMER_1m`, `TIMER_30s`, `VOLUME_DOWN`, `VOLUME_UP`, `WEATHER`.
- **Negatives (FAR):** put non-command clips in a folder named one of:
  `_negative`, `negatives`, `negative`, `unknown`, `background`, `noise`,
  `noncommand`, `non_command`. Example: wake-word clips, other speech, TV, silence,
  room noise — things that should be rejected as `UNKNOWN`.
- Aim for **~10–30 clips per intent** (more is better). Clips should be ~1.5 s and
  captured with the **same mic / distance** you'll use in the demo.
- You can also use a `manifest.csv` instead of folders:
  ```csv
  path,label
  clips/play1.wav,PLAY_MUSIC
  clips/weather1.wav,WEATHER
  clips/bg1.wav,_negative
  ```
  (labels starting with a negative name are treated as negatives)

Alternative locations the notebook also scans (first match wins):
`test_data/` here, then `../data/command/test/`, `../data/command/`, `../test_data/`,
`../data/`. Or point `TEST_DATA_DIR` / `NEG_DIR` at a custom path in the notebook's
config cell.

The notebook prints `positives: N` / `negatives: M`, so you can confirm it found your data.
