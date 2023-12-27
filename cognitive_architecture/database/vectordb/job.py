class Job:
    def __init__(self, job_id, job_status=None, filename=None):
        self.job_id = job_id
        self.job_status = job_status
        self.filename = filename

    def __str__(self):
        attributes = []
        if self.job_id is not None:
            attributes.append(f"job_id: {self.job_id}")
        if self.job_status is not None:
            attributes.append(f"job_status: {self.job_status}")
        if self.filename is not None:
            attributes.append(f"filename: {self.filename}")
        return "Job(" + ", ".join(attributes) + ")"

    def __repr__(self):
        return self.__str__()