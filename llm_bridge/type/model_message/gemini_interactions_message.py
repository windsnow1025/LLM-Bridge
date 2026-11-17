from google.genai.interactions import ContentParam, UserInputStepParam, ModelOutputStepParam

type GeminiInteractionsContent = ContentParam

type GeminiInteractionsMessage = UserInputStepParam | ModelOutputStepParam
