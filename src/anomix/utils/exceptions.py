# Generic Dataset based errors

# not implemented


# API User Interface Error ################################

class UserInputError(Exception):
    status_code = 400
    severity = 'low'

    def __init__(self, message, **kwargs):
        super().__init__(message)
        self.message = message
        self.details = kwargs


    def to_dict(self):
        return {'message': self.message,
                'type': "InputError",
                'severity': self.severity,
                'details': self.details}

class DataError(UserInputError):
    def __init__(self, message, **kwargs):
        super().__init__(message, **kwargs)

# Train Time Errors ###########################################################

class ModelNotFit(Exception):
    status_code = 500

    def __init__(self, reason, model_type):
        message = model_type + ' ' + reason
        super().__init__(message)
        self.message = message


class ModelFailedToFit(ModelNotFit):
    def __init__(self, model_type):
        super().__init__(reason='UnkownError', model_type=model_type)


class SingularMatrix(ModelNotFit):
    def __init__(self, model_type):
        super().__init__(reason='SingluarMatrix', model_type=model_type)


class SampleSizeTooSmall(ModelNotFit):
    def __init__(self, model_type):
        super().__init__(reason='SampleSizeTooSmall', model_type=model_type)




class NotEnoughVariation(ModelNotFit):
    def __init__(self, model_type):
        super().__init__(reason='NotEnoughVariation', model_type=model_type)


### Specific Likelihood Errors

class LikelihoodError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)


class LikelihoodDecreasingError(LikelihoodError):
    def __init__(self, message):
        super().__init__('Likelihood is Decreasing: '+ message)


class LikelihoodNaNError(LikelihoodError):
    def __init__(self):
        super().__init__('Error: NaN in likelihood')


class AllZeros(ModelNotFit):
    def __init__(self, model_type):
        super().__init__(reason='AllDataZero', model_type=model_type)