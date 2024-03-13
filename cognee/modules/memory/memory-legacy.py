class Memory:
    def __init__(
        self,
        user_id: str = "676",
        session=None,
        index_name: str = None,
        db_type: str = globalConfig.vectordb,
        namespace: str = None,
        memory_id: str = None,
        memory_class=None,
        job_id: str = None,
    ) -> None:
        self.load_environment_variables()
        self.memory_id = memory_id
        self.user_id = user_id
        self.session = session
        self.index_name = index_name
        self.db_type = db_type
        self.namespace = namespace
        self.memory_instances = []
        self.memory_class = memory_class
        self.job_id = job_id
        # self.memory_class = DynamicBaseMemory(
        #     "Memory", user_id, str(self.memory_id), index_name, db_type, namespace
        # )

    def load_environment_variables(self) -> None:
        self.OPENAI_TEMPERATURE = globalConfig.openai_temperature
        self.OPENAI_API_KEY = globalConfig.openai_key

    @classmethod
    async def create_memory(
        cls,
        user_id: str,
        session,
        job_id: str = None,
        memory_label: str = None,
        **kwargs,
    ):
        """
        Class method that acts as a factory method for creating Memory instances.
        It performs necessary DB checks or updates before instance creation.
        """
        existing_user = await cls.check_existing_user(user_id, session)
        logging.info(f"Existing user: {existing_user}")

        if existing_user:
            # Handle existing user scenario...
            memory_id = await cls.check_existing_memory(user_id, memory_label, session)
            if memory_id is None:
                memory_id = await cls.handle_new_memory(
                    user_id=user_id,
                    session=session,
                    job_id=job_id,
                    memory_name=memory_label,
                )
            logging.info(
                f"Existing user {user_id} found in the DB. Memory ID: {memory_id}"
            )
        else:
            # Handle new user scenario...
            await cls.handle_new_user(user_id, session)

            memory_id = await cls.handle_new_memory(
                user_id=user_id,
                session=session,
                job_id=job_id,
                memory_name=memory_label,
            )
            logging.info(
                f"New user {user_id} created in the DB. Memory ID: {memory_id}"
            )

        memory_class = DynamicBaseMemory(
            memory_label,
            user_id,
            str(memory_id),
            index_name=memory_label,
            db_type=globalConfig.vectordb,
            **kwargs,
        )

        return cls(
            user_id=user_id,
            session=session,
            memory_id=memory_id,
            job_id=job_id,
            memory_class=memory_class,
            **kwargs,
        )

    async def list_memory_classes(self):
        """
        Lists all available memory classes in the memory instance.
        """
        # Use a list comprehension to filter attributes that end with '_class'
        return [attr for attr in dir(self) if attr.endswith("_class")]

    @staticmethod
    async def check_existing_user(user_id: str, session):
        """Check if a user exists in the DB and return it."""
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def check_existing_memory(user_id: str, memory_label: str, session):
        """Check if a user memory exists in the DB and return it. Filters by user and label"""
        try:
            result = await session.execute(
                select(MemoryModel.id)
                .where(MemoryModel.user_id == user_id)
                .filter_by(memory_name=memory_label)
                .order_by(MemoryModel.created_at)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return None

    @staticmethod
    async def handle_new_user(user_id: str, session):
        """
        Handle new user creation in the database.

        Args:
            user_id (str): The unique identifier for the new user.
            session: The database session for the operation.

        Returns:
            str: A success message or an error message.

        Raises:
            Exception: If any error occurs during the user creation process.
        """
        try:
            new_user = User(id=user_id)
            await add_entity(session, new_user)
            return "User creation successful."
        except Exception as e:
            return f"Error creating user: {str(e)}"

    @staticmethod
    async def handle_new_memory(
        user_id: str,
        session,
        job_id: str = None,
        memory_name: str = None,
        memory_category: str = "PUBLIC",
    ):
        """
        Handle new memory creation associated with a user.

        Args:
            user_id (str): The user's unique identifier.
            session: The database session for the operation.
            job_id (str, optional): The identifier of the associated job, if any.
            memory_name (str, optional): The name of the memory.

        Returns:
            str: The unique memory ID if successful, or an error message.

        Raises:
            Exception: If any error occurs during memory creation.
        """
        try:
            memory_id = str(uuid.uuid4())
            logging.info("Job id %s", job_id)
            memory = MemoryModel(
                id=memory_id,
                user_id=user_id,
                operation_id=job_id,
                memory_name=memory_name,
                memory_category=memory_category,
                methods_list=str(["Memory", "SemanticMemory", "EpisodicMemory"]),
                attributes_list=str(
                    [
                        "user_id",
                        "index_name",
                        "db_type",
                        "knowledge_source",
                        "knowledge_type",
                        "memory_id",
                        "long_term_memory",
                        "short_term_memory",
                        "namespace",
                    ]
                ),
            )
            await add_entity(session, memory)
            return memory_id
        except Exception as e:
            return f"Error creating memory: {str(e)}"

    async def add_memory_instance(self, memory_class_name: str):
        """Add a new memory instance to the memory_instances list."""
        instance = DynamicBaseMemory(
            memory_class_name,
            self.user_id,
            self.memory_id,
            self.index_name,
            self.db_type,
            self.namespace,
        )
        print("The following instance was defined", instance)
        self.memory_instances.append(instance)

    async def query_method(self):
        methods_list = await self.session.execute(
            select(MemoryModel.methods_list).where(MemoryModel.id == self.memory_id)
        )
        methods_list = methods_list.scalar_one_or_none()
        return methods_list

    async def manage_memory_attributes(self, existing_user):
        """Manage memory attributes based on the user existence."""
        if existing_user:
            print(f"ID before query: {self.memory_id}, type: {type(self.memory_id)}")

            # attributes_list = await self.session.query(MemoryModel.attributes_list).filter_by(id=self.memory_id[0]).scalar()
            attributes_list = await self.query_method()
            logging.info(f"Attributes list: {attributes_list}")
            if attributes_list is not None:
                attributes_list = ast.literal_eval(attributes_list)
                await self.handle_attributes(attributes_list)
            else:
                logging.warning("attributes_list is None!")
        else:
            attributes_list = [
                "user_id",
                "index_name",
                "db_type",
                "knowledge_source",
                "knowledge_type",
                "memory_id",
                "long_term_memory",
                "short_term_memory",
                "namespace",
            ]
            await self.handle_attributes(attributes_list)

    async def handle_attributes(self, attributes_list):
        """Handle attributes for existing memory instances."""
        for attr in attributes_list:
            await self.memory_class.add_attribute(attr)

    async def manage_memory_methods(self, existing_user):
        """
        Manage memory methods based on the user existence.
        """
        if existing_user:
            # Fetch existing methods from the database
            # methods_list = await self.session.query(MemoryModel.methods_list).filter_by(id=self.memory_id).scalar()

            methods_list = await self.session.execute(
                select(MemoryModel.methods_list).where(
                    MemoryModel.id == self.memory_id[0]
                )
            )
            methods_list = methods_list.scalar_one_or_none()
            methods_list = ast.literal_eval(methods_list)
        else:
            # Define default methods for a new user
            methods_list = [
                "async_create_long_term_memory",
                "async_init",
                "add_memories",
                "fetch_memories",
                "delete_memories",
                "async_create_short_term_memory",
                "_create_buffer_context",
                "_get_task_list",
                "_run_main_buffer",
                "_available_operations",
                "_provide_feedback",
            ]
        # Apply methods to memory instances
        for class_instance in self.memory_instances:
            for method in methods_list:
                class_instance.add_method(method)

    async def dynamic_method_call(
        self, dynamic_base_memory_instance, method_name: str, *args, **kwargs
    ):
        if method_name in dynamic_base_memory_instance.methods:
            method = getattr(dynamic_base_memory_instance, method_name, None)
            if method:
                return await method(*args, **kwargs)
        raise AttributeError(
            f"{dynamic_base_memory_instance.name} object has no attribute {method_name}"
        )

    async def add_dynamic_memory_class(self, class_name: str, namespace: str):
        logging.info("Here is the memory id %s", self.memory_id[0])
        new_memory_class = DynamicBaseMemory(
            class_name,
            self.user_id,
            self.memory_id[0],
            self.index_name,
            self.db_type,
            namespace,
        )
        setattr(self, f"{class_name.lower()}_class", new_memory_class)
        return new_memory_class

    async def add_attribute_to_class(self, class_instance, attribute_name: str):
        # add this to database  for a particular user  and load under memory id
        await class_instance.add_attribute(attribute_name)

    async def add_method_to_class(self, class_instance, method_name: str):
        # add this to database for a particular user and load under memory id
        await class_instance.add_method(method_name)
