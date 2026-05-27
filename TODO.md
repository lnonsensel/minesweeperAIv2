# MinesweeperAIv2 — План улучшений

Проект реализует агента на базе Deep Q-Network (DQN), обучающегося играть в сапёр.
Среда написана на Gymnasium + Pygame, нейросеть на PyTorch.

---

## Критические баги

- [x] **`agent/dqn.py:38`** — `torch.cuda.is_available` вызывался без скобок (`is_available` вместо `is_available()`), устройство всегда определялось как `cpu`
- [x] **`minesweeper_env/game/minesweeper_game.py:45`** — `generate_fields` передавал `(self.field_size, self.field_size)` как кортеж кортежей вместо `self.field_size`
- [x] **`minesweeper_env/game/minesweeper_game.py`** — в `_right_click_action` флаг корректно проверяется через `opened_field[coords] == -1.` (момент установки) и `field[coords] == 1.` (raw-значение мины)
- [x] **`teacher/teacher.py:56`** — в `evaluate()` переменная `done` не сбрасывалась перед каждым эпизодом, оценка запускалась только для первой игры
- [x] **`teacher/teacher.py:103,108`** — `s = s_prime` присваивалось дважды подряд (лишняя строка после `else`)
- [x] **`agent/cnn.py:31-34`** — переменная `windows` перезаписывалась трижды, результаты `permute` и первого `view` терялись без использования
- [x] **`teacher/evaluator.py:27`** — `random.randint(0, eval_env.action_space.n)` включал `n` как допустимое значение, что вызывало `IndexError` (нужно `n - 1`)
- [x] **`minesweeper_env/minenv.py`** — в `get_reward()` мины сравнивались с `-3` вместо `1.` (raw-encoding), секция наград за флаги никогда не срабатывала
- [x] **`minesweeper_env/minenv.py`** — `(self.player_field == -1) & (self.field == -3)` всегда False; исправлено на `(self.opened_field == -1.) & (self.field == 1.)`
- [x] **`minesweeper_env/game/minesweeper_game.py`** — `last_action_type` и `last_action_coords` никогда не устанавливались в `play_action()`, секция наград за флаги в reward никогда не выполнялась

---

## Архитектура агента

- [x] **Векторизация `get_batch_channels`** — заменён Python-цикл на батчевые тензорные операции
- [x] **Gradient clipping** — добавлен `torch.nn.utils.clip_grad_norm_` перед `optimizer.step()`
- [x] **Double DQN** — online-сеть выбирает действие, target-сеть оценивает значение
- [x] **Soft target update** — `τ=0.005` каждый шаг вместо hard copy каждые 10k шагов
- [x] **Более глубокая CNN** — 3 Conv2d(3→32→64→64) + BatchNorm + ReLU + Conv1x1, receptive field 7×7
- [x] **Правильный flat-индекс в `learn()`** — action (type, y, x) конвертируется в позицию в Q-векторе
- [x] **Epsilon decay** — достигает `epsilon_min` за 150k шагов (75% бюджета)
- [x] **Dueling DQN** — выходной слой разделён на `Value` (scalar) и `Advantage` (2×H×W) потоки; Q = V + A − mean(A)
- [x] **Prioritized Experience Replay (PER)** — `SumTree` + `PrioritizedReplayBuffer`; семплирование пропорционально TD-ошибке, IS-взвешенный loss, beta annealing

---

## Функция награды

- [x] Убрать закомментированную старую версию `get_reward` в `minenv.py`
- [x] Вынести все константы наград в `RewardConfig` (`minesweeper_env/game/config.py`) для удобной настройки
- [ ] Добавить reward shaping на основе constraint satisfaction: если агент логически вычислил безопасную клетку — давать дополнительный бонус
- [x] Нормализовать награды — reward clipping через `RewardConfig.reward_clip` (0 = отключено)

---

## Логирование и мониторинг

- [x] **Win rate** — добавлена метрика процента выигранных партий в `evaluate()`
- [x] **Флаг `USE_LOGGER`** — вынесен в переменную окружения `MINESWEEPER_DEBUG`
- [x] Сохранение истории обучения в `evaluations/history.csv` рядом с графиком
- [x] `print(result)` → условный вывод только в режиме `info`
- [x] **TensorBoard** — логирование `train/loss`, `train/epsilon`, `eval/avg_return`, `eval/win_rate`; включается флагом `use_tensorboard` в `TeacherPreferences`

---

## Сохранение моделей

- [x] Сохранять лучшую модель по `eval score` (раньше — по `train reward`)
- [x] Сохранять полный checkpoint: веса сети, состояние оптимизатора, `total_steps`, `epsilon`, `top_eval_score`
- [x] `evaluator.py` поддерживает как новый формат dict, так и старые `.pt` файлы
- [x] Загрузка чекпоинта для продолжения обучения — поле `resume_from` в `TeacherPreferences`; GUI поддерживает ввод имени файла

---

## Качество кода

- [x] Переименовать `get_action_from_responce` → `get_action_from_response`
- [x] Удалить мёртвый дублирующий `get_channels` из `teacher/teacher.py`
- [x] `test.py` — переписан как настоящие unit-тесты с `pytest` (47 тестов, все проходят)
- [x] Удалить из `requirements.txt` зависимости, не относящиеся к проекту (`mujoco`, `box2d-py`, `ale-py`, `opencv-python`, `imageio`, `moviepy`, `gym` (old))
- [x] `is_left_click` явно приведён к `bool` в `play_action()`

---

## Среда и игровая логика

- [x] Добавлена проверка `mines_num < field_size[0] * field_size[1]` при инициализации среды
- [x] `last_action_type` и `last_action_coords` инициализируются в `reset_state()` и устанавливаются в `play_action()`
- [x] Задокументировано соглашение о порядке action_space в `ARCHITECTURE.md`: первые H×W — клики, следующие H×W — флаги

---

## Инфраструктура

- [x] Добавлен `README.md` с описанием проекта, инструкцией по установке и запуску
- [x] Добавлен `ARCHITECTURE.md` с подробным описанием всех модулей, кодирования поля, action space, reward function, CNN и training pipeline
- [x] Добавлен `Makefile` с командами `train`, `play`, `test`, `debug`
- [x] `.gitignore` — убран дубликат `.venv/`, добавлены `evaluations/*.png`, `evaluations/*.csv`, `.env`
- [x] Добавлен `pyproject.toml` с пакетной структурой
