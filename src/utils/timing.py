import time


def measure_training_time(
    pipeline,
    X_train,
    y_train
):
    """
    Measure model training time.
    """
    
    start = time.perf_counter()
    
    pipeline.fit(
        X_train,
        y_train
    )
    
    end = time.perf_counter()
    
    return end - start


def measure_prediction_time(
    pipeline,
    X_test
):
    """
    Measure model prediction time.
    """
    
    start = time.perf_counter()
    
    predictions = pipeline.predict(X_test)
    
    end = time.perf_counter()
    
    prediction_time = end - start
    
    return predictions, prediction_time