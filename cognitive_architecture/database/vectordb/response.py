from .job import Job


class Response:
    def __init__(self, error=None, message=None, successful_uploads=None, failed_uploads=None,
                 empty_files_count=None, duplicate_files_count=None, job_id=None,
                 jobs=None, job_status=None, status_code=None):
        self.error = error
        self.message = message
        self.successful_uploads = successful_uploads
        self.failed_uploads = failed_uploads
        self.empty_files_count = empty_files_count
        self.duplicate_files_count = duplicate_files_count
        self.job_id = job_id
        self.jobs = jobs
        self.job_status = job_status
        self.status_code = status_code

    @classmethod
    def from_json(cls, json_dict, status_code):
        successful_uploads = cls._convert_successful_uploads_to_jobs(json_dict.get('successful_uploads', None))
        jobs = cls._convert_to_jobs(json_dict.get('Jobs', None))

        return cls(
            error=json_dict.get('error'),
            message=json_dict.get('message'),
            successful_uploads=successful_uploads,
            failed_uploads=json_dict.get('failed_uploads'),
            empty_files_count=json_dict.get('empty_files_count'),
            duplicate_files_count=json_dict.get('duplicate_files_count'),
            job_id=json_dict.get('JobID'),
            jobs=jobs,
            job_status=json_dict.get('JobStatus'),
            status_code=status_code
        )

    @classmethod
    def _convert_successful_uploads_to_jobs(cls, successful_uploads):
        if not successful_uploads:
            return None
        return [Job(filename=key, job_id=val) for key, val in successful_uploads.items()]

    @classmethod
    def _convert_to_jobs(cls, jobs):
        if not jobs:
            return None
        return [Job(job_id=job['JobID'], job_status=job['JobStatus']) for job in jobs]

    def __str__(self):
        attributes = []
        if self.error is not None:
            attributes.append(f"error: {self.error}")
        if self.message is not None:
            attributes.append(f"message: {self.message}")
        if self.successful_uploads is not None:
            attributes.append(f"successful_uploads: {str(self.successful_uploads)}")
        if self.failed_uploads is not None:
            attributes.append(f"failed_uploads: {self.failed_uploads}")
        if self.empty_files_count is not None:
            attributes.append(f"empty_files_count: {self.empty_files_count}")
        if self.duplicate_files_count is not None:
            attributes.append(f"duplicate_files_count: {self.duplicate_files_count}")
        if self.job_id is not None:
            attributes.append(f"job_id: {self.job_id}")
        if self.jobs is not None:
            attributes.append(f"jobs: {str(self.jobs)}")
        if self.job_status is not None:
            attributes.append(f"job_status: {self.job_status}")
        if self.status_code is not None:
            attributes.append(f"status_code: {self.status_code}")

        return "Response(" + ", ".join(attributes) + ")"