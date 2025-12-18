await cognee.add(
    [
        """
        Harry Potter is a student at Hogwarts and belongs to Gryffindor house.
        He is known for defeating Voldemort and his Patronus is a stag.
        """,
        """
        Hermione Granger is a student at Hogwarts and also belongs to Gryffindor house.
        She is known for her intelligence and deep knowledge of spells. Her Patronus is an otter.
        """,
        """
        Severus Snape is a professor at Hogwarts who teaches Potions.
        He belongs to Slytherin house and was secretly loyal to Albus Dumbledore.
        """,
        """
        Hogwarts is a magical school located in Scotland. During Harry Potter's time at school, the headmaster was Albus Dumbledore.
        """,
        """
        A Horcrux is a dark magic object used to store a fragment of a wizard's soul. Voldemort created multiple Horcruxes to achieve immortality.
        """,
        """
        The Elder Wand is a powerful wand believed to be unbeatable. Its final known owner was Harry Potter.
        """,
    ],
    dataset_name="cognee-basics",
)
