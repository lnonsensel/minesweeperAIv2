# MinesweeperAIv2 — План улучшений

Проект реализует агента на базе Deep Q-Network (DQN), обучающегося играть в сапёр.
Среда написана на Gymnasium + Pygame, нейросеть на PyTorch.

---

## Критические баги

- [x] **`agent/dqn.py:38`** — `torch.cuda.is_available` вызывается без скобок (`is_available` вместо `is_available()`), устройство всегда определяется как `cpu`
- [x] **`minesweeper_env/game/minesweeper_game.py:45`** — `generate_fields` передаёт `(self.field_size, self.field_size)` как кортеж кортежей вместо `self.field_size`
- [x] **`minesweeper_env/game/minesweeper_game.py`** — в `_right_click_action` флаг корректно проверяется через `opened_field[coords] == -1.` (момент установки) и `field[coords] == 1.` (raw-значение мины)
- [x] **`teacher/teacher.py:56`** — в `evaluate()` переменная `done` не сбрасывается перед каждым эпизодом, оценка запускается только для первой игры
- [x] **`teacher/teacher.py:103,108`** — `s = s_prime` присваивается дважды подряд (лишняя строка после `else`)
- [x] **`agent/cnn.py:31-34`** — переменная `windows` перезаписывается трижды, результаты `permute` и первого `view` теряются без использования
- [x] **`teacher/evaluator.py:27`** — `random.randint(0, eval_env.action_space.n)` включает `n` как допустимое значение, что вызовет `IndexError` (нужно `n - 1`)
- [x] **`minesweeper_env/minenv.py`** — в `get_reward()` мины сравниваются с `-3` вместо `1.` (raw-encoding), секция наград за флаги никогда не срабатывала
- [x] **`minesweeper_env/minenv.py`** — `(self.player_field == -1) & (self.field == -3)` всегда False; исправлено на `(self.opened_field == -1.) & (self.field == 1.)`
- [x] **`minesweeper_env/game/minesweeper_game.py`** — `last_action_type` и `last_action_coords` никогда не устанавливались в `play_action()`, секция наград за флаги в reward никогда не выполнялась

---

## Архитектура агента

- [x] **Векторизация `get_batch_channels`** — заменён Python-цикл на батчевые тензорные операции
- [x] **Gradient clipping** — добавлен `torch.nn.utils.clip_grad_norm_` перед `optimizer.step()`
- [ ] **Double DQN** — текущий DQN переоценивает Q-значения; использовать основную сеть для выбора действия, целевую — для оценки
- [ ] **Dueling DQN** — разделить выходной слой на `Value` и `Advantage` потоки для более стабильного обучения
- [ ] **Prioritized Experience Replay (PER)** — семплировать переходы пропорционально TD-ошибке вместо равномерной выборки
- [ ] **Более глубокая CNN** — текущая архитектура (1 свёртка → LayerNorm → FC) слишком проста; добавить 2–3 свёрточных слоя с residual connections

---

## Функция награды

- [x] Убрать закомментированную старую версию `get_reward` в `minenv.py`
- [x] Вынести все константы наград в `RewardConfig` (`minesweeper_env/game/config.py`) для удобной настройки
- [ ] Добавить reward shaping на основе constraint satisfaction: если агент логически вычислил безопасную клетку — давать дополнительный бонус
- [ ] Нормализовать награды (reward clipping или running mean/std) для стабилизации обучения

---

## Логирование и мониторинг

- [x] **Win rate** — добавлена метрика процента выигранных партий в `evaluate()`
- [x] **Флаг `USE_LOGGER`** — вынесен в переменную окружения `MINESWEEPER_DEBUG`
- [x] Сохранение истории обучения в `evaluations/history.csv` рядом с графиком
- [x] `print(result)` → условный вывод только в режиме `info`
- [ ] **TensorBoard / wandb** — логирование метрик (loss, epsilon, eval score, win rate)

---

## Сохранение моделей

- [x] Сохранять лучшую модель по `eval score` (раньше — по `train reward`)
- [x] Сохранять полный checkpoint: веса сети, состояние оптимизатора, `total_steps`, `epsilon`, `top_eval_score`
- [x] `evaluator.py` поддерживает как новый формат dict, так и старые `.pt` файлы
- [ ] Добавить загрузку чекпоинта для продолжения обучения (`--resume` флаг)

---

## Качество кода

- [x] Переименовать `get_action_from_responce` → `get_action_from_response`
- [x] Удалить мёртвый дублирующий `get_channels` из `teacher/teacher.py`
- [x] `test.py` — переписан как настоящие unit-тесты с `pytest` (28 тестов, все проходят)
- [x] Удалить из `requirements.txt` зависимости, не относящиеся к проекту (`mujoco`, `box2d-py`, `ale-py`, `opencv-python`, `imageio`, `moviepy`, `gym` (old))
- [x] `is_left_click` явно приведён к `bool` в `play_action()`

---

## Среда и игровая логика

- [x] Добавлена проверка `mines_num < field_size[0] * field_size[1]` при инициализации среды
- [x] `last_action_type` и `last_action_coords` инициализируются в `reset_state()` и устанавливаются в `play_action()`
- [ ] Задокументировать соглашение о порядке action_space: первые H×W — клики, следующие H×W — флаги

---

## Инфраструктура

- [x] Добавлен `README.md` с описанием проекта, инструкцией по установке и запуску
- [x] Добавлен `Makefile` с командами `train`, `play`, `test`, `debug`
- [x] `.gitignore` — убран дубликат `.venv/`, добавлены `evaluations/*.png`, `evaluations/*.csv`, `.env`
- [ ] Добавить `pyproject.toml` с пакетной структурой
