# CIFAR10_vgg16
## Creating a model for image classification using the VGG16 network. / Создании модели для классификации изображений с использованием сети VGG16.

Цель: Создание модели для распознавания изображений с использованием сети VGG16 и методов transfer learning и fine-tuning. Я стремился к созданию эффективной модели, способной классифицировать изображения на 10 различных классов. После этого я планировал провести fine-tuning модели, чтобы улучшить ее производительность на конкретной задаче классификации

Результат: После применения методов мне удалось получить модель, которая показала хорошие результаты в классификации изображений на 10 классов. Благодаря использованию предварительно обученной модели и дальнейшей настройке на новом датасете удалось достичь высокой точности распознавания.

Стек технологий: Для решения задачи я использовал библиотеки глубокого обучения TensorFlow, Keras, Pandas. Также для предварительной обработки данных и визуализации результатов я использовал библиотеки numpy, matplotlib и seaborn.

[В работе использовался датасет CIFAR-10 из соревнований Kaggle](https://www.kaggle.com/c/cifar-10)

## VGG16 с тонкой настройкой, основные моменты:
* Заморозка опорной сети (первые 18 слоев), используется небольшая скорость обучения для тонкой настройки классификационного заголовка, начальная скорость обучения lr=0.0001, оптимизатор RMSprop, и тонкая настройка 15/20 эпох.
* Далее была реализована частичная сеть и тонкая настройка классификатора.

## VGG16 finetuning, основные моменты:
* Загрузить VGG16 без верхнего слоя классификации и подготовьть пользовательский классификатор.
* Наложить обе модели друг на друга.
* Также был использован класс EarlyStopping. Он реализует функционал ранней остаоновки, так мы сможем избежать переобучения нейросети.

### Выход сети с предсказаниями классов / Network output with class predictions
![image](https://github.com/ArtemAvgutin/CIFAR10_vgg16/assets/131138862/0133ae8a-bacd-4b85-98e0-733853b18add)

### График обчуения и ранней остановки при переобучении / Schedule for learning and early stopping during retraining
![image](https://github.com/ArtemAvgutin/CIFAR10_vgg16/assets/131138862/0f65630f-f3c1-4c6f-904f-c68b7258e55e)

Goal: Creating a model for image recognition using the VGG16 network and transfer learning and fine-tuning methods. I aimed to create an efficient model that could classify images into 10 different classes. After this, I planned to fine-tun the model to improve its performance on a specific classification task.

Result: After applying the methods, I was able to obtain a model that showed good results in classifying images into 10 classes. Thanks to the use of a pre-trained model and further adjustment on the new dataset, it was possible to achieve high recognition accuracy.

Technology stack: To solve the problem, I used the deep learning libraries TensorFlow, Keras, Pandas. I also used the numpy, matplotlib and seaborn libraries to preprocess the data and visualize the results.

[The work used the CIFAR-10 dataset from the Kaggle competition](https://www.kaggle.com/c/cifar-10)

## VGG16 with fine tuning, highlights:
* Freeze the core network (first 18 layers), use a small learning rate to fine-tune the classification header, initial learning rate lr=0.0001, RMSprop optimizer, and fine-tune 15/20 epochs.
* Next, a partial network and fine-tuning of the classifier were implemented.

## VGG16 finetuning, highlights:
* Load VGG16 without the top classification layer and prepare a custom classifier.
* Superimpose both models on top of each other.
* The EarlyStopping class was also used. It implements the early stopping functionality, so we can avoid retraining the neural network.
