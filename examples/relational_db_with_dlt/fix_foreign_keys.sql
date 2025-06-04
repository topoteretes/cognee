BEGIN TRANSACTION;

PRAGMA foreign_keys=OFF;

/*------------------------------------------------------------------------
1) pokemon_list
------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS new_pokemon_list (
    name TEXT,
    url TEXT,
    _dlt_load_id TEXT NOT NULL,
    _dlt_id VARCHAR(128) NOT NULL PRIMARY KEY
);
INSERT INTO new_pokemon_list
    SELECT name, url, _dlt_load_id, _dlt_id
    FROM pokemon_list;
DROP TABLE pokemon_list;
ALTER TABLE new_pokemon_list RENAME TO pokemon_list;

/*------------------------------------------------------------------------
2) pokemon_details (Parent for most child tables)
   _dlt_id is a PRIMARY KEY so children can FK to it.
------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS new_pokemon_details (
    base_experience BIGINT,
    height BIGINT,
    id BIGINT,
    is_default BOOLEAN,
    name TEXT,
    "order" BIGINT,
    species__name,
    weight BIGINT,
    _dlt_load_id TEXT NOT NULL,
    _dlt_id VARCHAR(128) NOT NULL PRIMARY KEY
);
INSERT INTO new_pokemon_details
    SELECT base_experience,
           height,
           id,
           is_default,
           name,
           "order",
           species__name,
           weight,
           _dlt_load_id,
           _dlt_id
    FROM pokemon_details;
DROP TABLE pokemon_details;
ALTER TABLE new_pokemon_details RENAME TO pokemon_details;

/*------------------------------------------------------------------------
3) pokemon_details_abilities (Child)
   Foreign key from _dlt_parent_id → pokemon_details(_dlt_id)
------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS new_pokemon_details__abilities (
    ability__name TEXT,
    ability__url TEXT,
    is_hidden BOOLEAN,
    slot BIGINT,
    _dlt_parent_id VARCHAR(128) NOT NULL,
    _dlt_list_idx BIGINT NOT NULL,
    _dlt_id VARCHAR(128) NOT NULL PRIMARY KEY,
    CONSTRAINT fk_abilities
      FOREIGN KEY (_dlt_parent_id)
      REFERENCES pokemon_details(_dlt_id)
      ON DELETE CASCADE
);
INSERT INTO new_pokemon_details__abilities
    SELECT ability__name,
           ability__url,
           is_hidden,
           slot,
           _dlt_parent_id,
           _dlt_list_idx,
           _dlt_id
    FROM pokemon_details__abilities;
DROP TABLE pokemon_details__abilities;
ALTER TABLE new_pokemon_details__abilities RENAME TO pokemon_details__abilities;

/*------------------------------------------------------------------------
4) pokemon_details_forms (Child)
------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS new_pokemon_details_forms (
    name TEXT,
    url TEXT,
    _dlt_parent_id VARCHAR(128) NOT NULL,
    _dlt_list_idx BIGINT NOT NULL,
    _dlt_id VARCHAR(128) NOT NULL PRIMARY KEY,
    FOREIGN KEY (_dlt_parent_id)
      REFERENCES pokemon_details(_dlt_id)
      ON DELETE CASCADE
);
INSERT INTO new_pokemon_details_forms
    SELECT name,
           url,
           _dlt_parent_id,
           _dlt_list_idx,
           _dlt_id
    FROM pokemon_details__forms;
DROP TABLE pokemon_details__forms;
ALTER TABLE new_pokemon_details_forms RENAME TO pokemon_details__forms;

/*------------------------------------------------------------------------
5) pokemon_details_game_indices (Child)
------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS new_pokemon_details__game_indices (
    game_index BIGINT,
    version__name TEXT,
    version__url TEXT,
    _dlt_parent_id VARCHAR(128) NOT NULL,
    _dlt_list_idx BIGINT NOT NULL,
    _dlt_id VARCHAR(128) NOT NULL PRIMARY KEY,
    FOREIGN KEY (_dlt_parent_id)
      REFERENCES pokemon_details(_dlt_id)
      ON DELETE CASCADE
);
INSERT INTO new_pokemon_details__game_indices
    SELECT game_index,
           version__name,
           version__url,
           _dlt_parent_id,
           _dlt_list_idx,
           _dlt_id
    FROM pokemon_details__game_indices;
DROP TABLE pokemon_details__game_indices;
ALTER TABLE new_pokemon_details__game_indices RENAME TO pokemon_details__game_indices;


/*------------------------------------------------------------------------
6) pokemon_details_moves (Child of pokemon_details)
------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS new_pokemon_details__moves (
    move__name TEXT,
    move__url TEXT,
    _dlt_parent_id VARCHAR(128) NOT NULL,
    _dlt_list_idx BIGINT NOT NULL,
    _dlt_id VARCHAR(128) NOT NULL PRIMARY KEY,
    FOREIGN KEY (_dlt_parent_id)
      REFERENCES pokemon_details(_dlt_id)
      ON DELETE CASCADE
);
INSERT INTO new_pokemon_details__moves
    SELECT move__name,
           move__url,
           _dlt_parent_id,
           _dlt_list_idx,
           _dlt_id
    FROM pokemon_details__moves;
DROP TABLE pokemon_details__moves;
ALTER TABLE new_pokemon_details__moves RENAME TO pokemon_details__moves;

/*------------------------------------------------------------------------
7) pokemon_details_moves_version_group_details (Child of pokemon_details_moves)
------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS new_pokemon_details__moves__version_group_details (
    level_learned_at BIGINT,
    version_group__name TEXT,
    version_group__url TEXT,
    move_learn_method__name TEXT,
    move_learn_method__url TEXT,
    _dlt_parent_id VARCHAR(128) NOT NULL,
    _dlt_list_idx BIGINT NOT NULL,
    _dlt_id VARCHAR(128) NOT NULL PRIMARY KEY,
    "order" BIGINT,
    FOREIGN KEY (_dlt_parent_id)
      REFERENCES pokemon_details__moves(_dlt_id)
      ON DELETE CASCADE
);
INSERT INTO new_pokemon_details__moves__version_group_details
    SELECT level_learned_at,
           version_group__name,
           version_group__url,
           move_learn_method__name,
           move_learn_method__url,
           _dlt_parent_id,
           _dlt_list_idx,
           _dlt_id,
           "order"
    FROM pokemon_details__moves__version_group_details;
DROP TABLE pokemon_details__moves__version_group_details;
ALTER TABLE new_pokemon_details__moves__version_group_details
    RENAME TO pokemon_details__moves__version_group_details;

/*------------------------------------------------------------------------
8) pokemon_details_past_abilities (Child of pokemon_details)
------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS new_pokemon_details__past_abilities (
    generation__name TEXT,
    generation__url TEXT,
    _dlt_parent_id VARCHAR(128) NOT NULL,
    _dlt_list_idx BIGINT NOT NULL,
    _dlt_id VARCHAR(128) NOT NULL PRIMARY KEY,
    FOREIGN KEY (_dlt_parent_id)
      REFERENCES pokemon_details(_dlt_id)
      ON DELETE CASCADE
);
INSERT INTO new_pokemon_details__past_abilities
    SELECT generation__name,
           generation__url,
           _dlt_parent_id,
           _dlt_list_idx,
           _dlt_id
    FROM pokemon_details__past_abilities;
DROP TABLE pokemon_details__past_abilities;
ALTER TABLE new_pokemon_details__past_abilities
    RENAME TO pokemon_details__past_abilities;

/*------------------------------------------------------------------------
9) pokemon_details_stats (Child of pokemon_details)
------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS new_pokemon_details_stats (
    base_stat BIGINT,
    effort BIGINT,
    stat__name TEXT,
    stat__url TEXT,
    _dlt_parent_id VARCHAR(128) NOT NULL,
    _dlt_list_idx BIGINT NOT NULL,
    _dlt_id VARCHAR(128) NOT NULL PRIMARY KEY,
    FOREIGN KEY (_dlt_parent_id)
      REFERENCES pokemon_details(_dlt_id)
      ON DELETE CASCADE
);
INSERT INTO new_pokemon_details_stats
    SELECT base_stat,
           effort,
           stat__name,
           stat__url,
           _dlt_parent_id,
           _dlt_list_idx,
           _dlt_id
    FROM pokemon_details__stats;
DROP TABLE pokemon_details__stats;
ALTER TABLE new_pokemon_details_stats
    RENAME TO pokemon_details__stats;

/*------------------------------------------------------------------------
10) pokemon_details_types (Child of pokemon_details)
------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS new_pokemon_details__types (
    slot BIGINT,
    type__name TEXT,
    type__url TEXT,
    _dlt_parent_id VARCHAR(128) NOT NULL,
    _dlt_list_idx BIGINT NOT NULL,
    _dlt_id VARCHAR(128) NOT NULL PRIMARY KEY,
    FOREIGN KEY (_dlt_parent_id)
      REFERENCES pokemon_details(_dlt_id)
      ON DELETE CASCADE
);
INSERT INTO new_pokemon_details__types
    SELECT slot,
           type__name,
           type__url,
           _dlt_parent_id,
           _dlt_list_idx,
           _dlt_id
    FROM pokemon_details__types;
DROP TABLE pokemon_details__types;
ALTER TABLE new_pokemon_details__types
    RENAME TO pokemon_details__types;

/*------------------------------------------------------------------------
11) pokemon_details_past_abilities_abilities (Child of pokemon_details_past_abilities)
------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS new_pokemon_details__past_abilities__abilities (
    is_hidden BOOLEAN,
    slot BIGINT,
    _dlt_parent_id VARCHAR(128) NOT NULL,
    _dlt_list_idx BIGINT NOT NULL,
    _dlt_id VARCHAR(128) NOT NULL PRIMARY KEY,
    FOREIGN KEY (_dlt_parent_id)
      REFERENCES pokemon_details__past_abilities(_dlt_id)
      ON DELETE CASCADE
);
INSERT INTO new_pokemon_details__past_abilities__abilities
    SELECT is_hidden,
           slot,
           _dlt_parent_id,
           _dlt_list_idx,
           _dlt_id
    FROM pokemon_details__past_abilities__abilities;
DROP TABLE pokemon_details__past_abilities__abilities;
ALTER TABLE new_pokemon_details__past_abilities__abilities
    RENAME TO pokemon_details__past_abilities__abilities;


/* Re-enable FK checks */
PRAGMA foreign_keys=ON;

COMMIT;
