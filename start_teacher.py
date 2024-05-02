from teacher.preferences import TeacherPreferences
from teacher.teacher import Teacher
TeacherPreferences()
Teacher()
def start_teacher(preferences: TeacherPreferences):
    teacher = Teacher(agent_preferences=preferences.agent_preferences,
                      env_preferences=preferences.env_preferences,
                      eval_interval=preferences.eval_interval,
                      learning_max_steps=preferences.learning_max_steps)