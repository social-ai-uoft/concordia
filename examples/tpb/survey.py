import datetime

from collections.abc import Callable

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component

from examples.tpb.agent import AgentConfig

# Introductory texts
MANIPULATIONS = {
  "attitude": """
Today, we're going to talk about the numerous benefits of regular physical activity.

Regular physical activity isn't just good for your body; it's great for your overall well-being. Studies have shown that staying active can improve your health, boost your mood, reduce stress, and enhance your cognitive abilities.

Moreover, students who attend our Sports and Recreation (S&R) services are more likely to maintain physical activity over the term compared to those who don't. This is based on evidence from a larger ongoing study, indicating that our program fosters a supportive environment conducive to sustained physical activity.

Safety is paramount to us. Our S&R facilities are equipped with comprehensive induction programs, continuous monitoring, and trained support staff to ensure your safety and minimize injury risks during physical activities.

Additionally, we offer a wide range of exercise classes and individual activities tailored to suit your preferences and fitness levels. Our flexible timetable allows you to customize your program according to your availability and interests using our user-friendly website.

So, whether you're into swimming, hitting the fitness suite, playing squash, or trying out other activities, we've got you covered. Let's embark on this journey towards a healthier, happier you together!
""",
  "norm": """
Today, we're going to discuss the importance of social approval and support in fostering physical activity participation.

Many students report that not having someone to actively participate in physical activity with them is a significant barrier to becoming more active. However, students who attend our Sports and Recreation (S&R) services often find that they meet like-minded individuals who not only approve but also actively support and participate in physical activity with them. This creates a supportive environment where you can feel encouraged and motivated to stay active.

It's essential to recognize that most friends and family actually approve of involvement in physical activity. They understand the importance of staying active for a safe, secure, and healthy lifestyle. By engaging in physical activity, you're not only benefiting yourself but also setting a positive example for those around you.

Misperceptions about others' disapproval for involvement in regular physical activity often stem from competing expectations about time management rather than disapproval for the activity itself. Participants in our pilot study found that addressing potential time conflicts and explicitly communicating with important others can often turn misperceptions into approval and support.

Moreover, it's crucial to debunk the misconception that regular S&R users are elitist super-athletes who look down on others. In reality, users of our services represent a diverse cross-section of the university community. Whether you're a beginner or an experienced athlete, everyone is welcome at our facilities.

Join us at the S&R services and experience the positive impact of social approval and support on your physical activity journey!
""",
  "control": """
Today, we're going to address the key barriers to participation in physical activity and how we've taken steps to overcome them at our Sports and Recreation (S&R) facilities.

Firstly, let's talk about costs. Our pilot studies have revealed that students often overestimate the costs for admission. In reality, the cost of entry is only GBP£1.60 (approximately USD$2.65) for a single visit, making it affordable for everyone. Additionally, we offer an annual membership for students at just GBP£82 (approximately USD$135), providing excellent value for money and eliminating financial barriers to participation.

Next, let's tackle the issue of time constraints. We understand that students lead busy lives, which is why we emphasize our long and flexible opening hours. Our facilities are open before, between, and after lectures, as well as during lunchtime. With over 35 classes each week throughout the day, every day, you can easily find a time that fits your schedule. Plus, our user-friendly website allows you to plan your workouts in advance, ensuring that you can prioritize physical activity without compromising your other commitments.

Access to our facilities is another important consideration. We've made it easy for students to reach us by providing detailed information about our locations, parking options, and public transport routes from the university's remote hospital campus. Whether you're commuting from campus or elsewhere, getting to our facilities is convenient and hassle-free.

Finally, let's address feelings of discomfort and embarrassment about exercising in public. We understand that this can be a significant barrier for some individuals. However, it's essential to recognize that the idea about the typical user of our facilities is often biased. Our users come from diverse backgrounds and fitness levels, creating a welcoming and inclusive environment where everyone feels comfortable to pursue their fitness goals.

Join us at the S&R services and experience firsthand how we've addressed these barriers to make physical activity accessible and enjoyable for all!
"""}

class Survey(component.Component):
  """Survey document for the TPB agent."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      config: AgentConfig,
      clock_now: Callable[[], datetime.datetime],
      interventions: tuple[str, ...],
      num_memories: int = 10
  ):
    self._model = model
    self._config = config
    self._interventions = interventions
    self._clock_now = clock_now
    self._num_memories = num_memories

  def name(self) -> str:
    return "Survey"

  def update(self):

    answers = []

    for intervention in self._interventions:
      prompt = interactive_document.InteractiveDocument(model=self._model)
      prompt.statement("Hello and thank you for participating in our study!")
      mems = "\n".join(self._config.memory.retrieve_recent(
        self._num_memories, add_time=True
      ))
      prompt.statement(f"Memories of {self._config.name}: {mems}")
      prompt.statement(f"Traits of {self._config.name}: {self._config.traits}")
      prompt.statement(f"Goals of {self._config.name}: {self._config.goal}")
      prompt.statement(f"Current time: {self._clock_now()}")
      prompt.statement(MANIPULATIONS[intervention])
      yn = prompt.yes_no_question("Were you aware of this? Just answer with (a) or (b), nothing else.")
      importance = prompt.open_question(
        "On a scale of 1 to 7, please rate how important this answer is for you. "
        "Provide only the number as the answer. Do not provide any explanation or "
        "description, just the number that represents how important this is for you."
      )
      answers.append({intervention: {"awareness": yn, "importance": importance}})
    
    self._answers = answers

  @property
  def answers(self):
    return self._answers