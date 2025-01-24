import os
import pathlib
from os.path import join
from typing import Any, Dict, List, Optional, Union
import re

import jinja2
from pydantic import BaseModel
import textx.scoping.providers as scoping_providers
from rich import pretty, print
from textx import (
    TextXSemanticError,
    get_children_of_type,
    language,
    metamodel_from_file,
)
from textx.scoping import GlobalModelRepository

import dflow.definitions as CONSTANTS

from dflow.generator import validate_path_params, process_eservice_params_as_dict
from dflow.m2m.openapi_to_dflow import Trigger
from dflow.similarity import are_lists_similar

pretty.install()

CURRENT_FPATH = pathlib.Path(__file__).parent.resolve()

GLOBAL_REPO = GlobalModelRepository()


def model_proc(model, metamodel):
    pass


CUSTOM_CLASSES = [
]


def class_provider(name):
    classes = dict(map(lambda x: (x.__name__, x), CUSTOM_CLASSES))
    return classes.get(name)


def component_processor(component):
    if component.attribute == None:
        component.attribute = ""
    return component

def nid_processor(nid):
    nid = nid.replace("\n", "")
    return nid


obj_processors = {
    'NID': nid_processor,
}


def get_metamodel(debug: bool = False, global_repo: bool = False):
    metamodel = metamodel_from_file(
        join(CONSTANTS.THIS_DIR, 'grammar', 'dflow.tx'),
        classes=class_provider,
        auto_init_attributes=True,
        textx_tools_support=True,
        # global_repository=GLOBAL_REPO,
        global_repository=global_repo,
        debug=debug,
    )

    # metamodel.register_scope_providers(get_scode_providers())
    metamodel.register_model_processor(model_proc)
    metamodel.register_obj_processors(obj_processors)
    return metamodel


def get_scode_providers():
    sp = {"*.*": scoping_providers.FQNImportURI(importAs=True)}
    if CONSTANTS.BUILTIN_MODELS:
        sp["brokers*"] = scoping_providers.FQNGlobalRepo(
            join(CONSTANTS.BUILTIN_MODELS, "broker", "*.dflow"))
        # sp["entities*"] = scoping_providers.FQNGlobalRepo(
        #     join(BUILTIN_MODELS, "entity", "*.dflow"))
    if CONSTANTS.MODEL_REPO_PATH:
        sp["brokers*"] = scoping_providers.FQNGlobalRepo(
            join(CONSTANTS.MODEL_REPO_PATH, "broker", "*.dflow"))
        # sp["entities*"] = scoping_providers.FQNGlobalRepo(
        #     join(MODEL_REPO_PATH, "entity", "*.dflow"))
    return sp


def has_duplicates(input_list):
    """ Returns weather the input_list has any duplicate values, and the duplicate value. """
    seen = set()
    for item in input_list:
        if item in seen:
            return True, item
        seen.add(item)
    return False, None


def _validate_model(model):
    """ Runs semantic validation on the provided model and raises Errors. """

    all_concept_names = []
    # Validate Intents
    intents = get_children_of_type("Intent", model)
    if len(intents) < 1:
        raise TextXSemanticError("There must be at least 1 Intent provided!")
    intents_names = [i.name for i in intents]
    check, _intent = has_duplicates(intents_names)
    if check:
        raise TextXSemanticError(f"Intents ID `{_intent}` is used multiple times!")
    all_concept_names.extend(intents_names)

    # Validate for at least 2 examples per intent
    for intent in intents:
        if len(intent.phrases) < 2:
            raise TextXSemanticError(f'Only {len(intent.phrases)} given in intent {intent}! At least 2 are needed!')

    # Validate Entities
    entities = get_children_of_type("TrainableEntity", model)
    entities_names = [e.name for e in entities]
    check, _entity = has_duplicates(entities_names)
    if check:
        raise TextXSemanticError(f"Entities ID '{_entity}' is used multiple times!")
    all_concept_names.extend(entities_names)

    # Validate trainable entities examples - no duplicates among different entities
    entities_words = [e.words for e in entities]
    entities_words = [set(_words) for _words in entities_words]

    seen = set()
    value_to_sets = {}
    # Iterate through each set in the list
    for idx, s in enumerate(entities_words):
        if not len(s):
            raise Exception(f'No examples given for entity {entities_words[idx]}.')
        for item in s:
            if item in seen:
                prev_idx = value_to_sets[item][0]
                raise TextXSemanticError(f"Trainable Entity examples must exist only in one entity. Entity example `{item}` found in entity `{entities_names[idx]}` and entity `{entities_names[prev_idx]}`!")
            else:
                seen.add(item)
                if item not in value_to_sets:
                    value_to_sets[item] = [idx]

    # Validate Pretrained Entities - they must have at least one example
    pretrained_entities_examples = {}
    for intent in intents:
        for complex_phrase in intent.phrases:
            for phrase in complex_phrase.phrases:
                if phrase.__class__.__name__ == "PretrainedEntityRef":
                    name = phrase.entity
                    if name not in pretrained_entities_examples:
                        pretrained_entities_examples[name] = []
                    if phrase.refPreValues != []:
                        pretrained_entities_examples[name].extend(phrase.refPreValues)
    for pe, examples in pretrained_entities_examples.items():
        if pe not in CONSTANTS.PE_CLASSES_LIST:
            raise TextXSemanticError(f"Pretrained Entity `{pe}` is not in the supported entity classes.")
        if not len(examples):
            raise TextXSemanticError(f"No example given for Pretrained Entity `{pe}`.")

    # Validate Synonyms
    synonyms = get_children_of_type("Synonym", model)
    synonyms_names = [s.name for s in synonyms]
    check, _synonym = has_duplicates(synonyms_names)
    if check:
        raise TextXSemanticError(f"Synonyms ID `{_synonym}` is used multiple times!")
    all_concept_names.extend(synonyms_names)

    synonyms_words = [e.words for e in synonyms]
    synonyms_words = [set(_words) for _words in synonyms_words]

    seen = set()
    value_to_sets = {}
    # Iterate through each set in the list
    for idx, s in enumerate(synonyms_words):
        if not len(s):
            raise Exception(f'No examples given for synonym {entities_words[idx]}.')
        for item in s:
            if item in seen:
                prev_idx = value_to_sets[item][0]
                raise TextXSemanticError(f"Synonym phrases must exist only in one synonym. Synonym phrase `{item}` found in synonym `{synonyms_names[idx]}` and synonym `{synonyms_names[prev_idx]}`!")
            else:
                seen.add(item)
                if item not in value_to_sets:
                    value_to_sets[item] = [idx]

    # Validate EServices
    eservices = get_children_of_type("EServiceDefHTTP", model)
    eservices_names = [e.name for e in eservices]
    check, _eservice = has_duplicates(eservices_names)
    if check:
        raise TextXSemanticError(f"EService ID `{_eservice}` is used multiple times!")
    all_concept_names.extend(eservices_names)

    eservices_info = {}
    for service in eservices:
        service_info = {}
        service_info['verb'] = service.verb
        service_info['host'] = service.host
        if service.port:
            service_info['port'] = service.port
            port = f":{service.port}"
        else:
            service_info['port'] = ''
            port = ''

        service_info['mime'] = ''
        if service.mime:
            if service.verb.lower() == 'get':
                for mime in service.mime:
                    service_info['mime'] += f"'Accept': '{mime}', "
            else:
                for mime in service.mime:
                    service_info['mime'] += f"'Content-Type': '{mime}', "

        service_info['path'] = service.path
        service_info['url'] = f"{service_info['host']}{port}{service_info['path']}"
        eservices_info[service.name] = service_info

    # Validate Dialogues
    dialogues = get_children_of_type("Dialogue", model)
    if not len(dialogues):
        raise TextXSemanticError("There must be at least 1 Dialogue!")

    dialogues_names = [d.name for d in dialogues]
    check, _dialogue = has_duplicates(dialogues_names)
    if check:
        raise TextXSemanticError(f"Dialogues ID `{_dialogue}` is used multiple times!")
    all_concept_names.extend(dialogues_names)

    responses_names = [resp.name for d in dialogues for resp in d.responses]
    check, _response = has_duplicates(responses_names)
    if check:
        raise TextXSemanticError(f"Responses ID `{_response}` is used multiple times!")
    all_concept_names.extend(responses_names)

    action_groups_names = []
    slot_names = []
    inline_policies_roles = []
    for dialogue in dialogues:
        for response in dialogue.responses:
            if response.__class__.__name__ == 'ActionGroup':
                action_groups_names.append(response.name)
                for action in response.actions:
                    if action.roles:
                        inline_policies_roles.extend(action.roles)
                    if action.__class__.__name__ == 'EServiceCallHTTP':
                        path_params, _, _, _ = process_eservice_params_as_dict(action.path_params)
                        validation = validate_path_params(eservices_info[action.eserviceRef.name]['url'], path_params)
                        if not validation:
                            raise Exception(f'Service `{action.eserviceRef.name}` path and path params do not match when called in `{response.name}`.')
            else:
                for slot in response.params:
                    slot_names.append(slot.name)
                    if slot.source.__class__.__name__ == 'EServiceCallHTTP':
                        path_params, _, _, _ = process_eservice_params_as_dict(slot.source.path_params)
                        validation = validate_path_params(eservices_info[slot.source.eserviceRef.name]['url'], path_params)
                        if not validation:
                            raise Exception(f'Service `{slot.source.eserviceRef.name}` path and path params do not match when called in `{response.name}` for slot `{slot.name}`.')

    # Validate Global Slots
    gslots = get_children_of_type("GlobalSlot", model)
    gslots_names = [gs.name for gs in gslots]
    check, _gslot = has_duplicates(gslots_names)
    if check:
        raise TextXSemanticError(f"GSlots ID `{_gslot}` is used multiple times!")
    all_concept_names.extend(gslots_names)

    slot_names.extend(gslots_names)

    # Validate Events
    events = get_children_of_type("Event", model)
    events_names = [e.name for e in events]
    check, _event = has_duplicates(events_names)
    if check:
        raise TextXSemanticError(f"Events ID `{_event}` is used multiple times!")
    all_concept_names.extend(events_names)

    # Validate Access Control
    access_control = get_children_of_type("AccessControlDef", model)
    if access_control:
        roles = access_control[0].roles.words
        check, _role = has_duplicates(roles)
        if check:
            raise TextXSemanticError(f"Duplicate role ID `{_role}`!")

        if not access_control[0].roles.default:
            raise TextXSemanticError(f"Default role is not defined under Roles!")

        path = access_control[0].path
        if path and not os.path.isfile(path.path):
            raise TextXSemanticError(f'File not found: {path.path}')

        users = access_control[0].users.roles
        if not (path or users):
            raise TextXSemanticError(f"Both 'Users' and 'Path' were not defined")

        _role_users = {}
        for role in access_control[0].users.roles:
            if role.role in _role_users.keys():
                raise TextXSemanticError(f"Duplicate role `{role.role}` in 'Users.")
            _role_users[role.role] = role.users

        policies = access_control[0].policies
        for policy in policies:
            for role in policy.roles:
                if role not in roles:
                    raise TextXSemanticError(f"Role `{role}` is not defined under Roles!")
            for action in policy.actions:
                if action not in action_groups_names:
                    raise TextXSemanticError(f"Action: `{action}` in Policy `{policy.name}` is not a defined ActionGroup")

        if access_control[0].authentication.method == 'slot':
            if not access_control[0].authentication.slot_name:
                raise TextXSemanticError("You need to provide a 'slot_name' for this authentication method")
            if access_control[0].authentication.slot_name not in slot_names:
                raise TextXSemanticError(f"Authentication slot `{access_control[0].authentication.slot_name}` not defined!")

        for role in inline_policies_roles:
            if role not in roles:
                raise TextXSemanticError(f"Role `{role}` defined in an inline policy is not defined under Roles!")

    # Check duplicates among all IDs
    check, _attributes = has_duplicates(all_concept_names)
    if check:
        raise TextXSemanticError(f"ID `{_attributes}` is used multiple times in different concepts!")
    return


def build_model(model_path: str, debug: bool = False):
    # Parse model
    mm = get_metamodel(debug=debug)
    model = mm.model_from_file(model_path)
    _validate_model(model)
    return model


def report_model_info(model):
    entities = get_children_of_type("TrainableEntity", model)
    synonyms = get_children_of_type("Synonym", model)
    gslots = get_children_of_type("GlobalSlot", model)
    intents = get_children_of_type("Intent", model)
    events = get_children_of_type("Event", model)
    eservices = get_children_of_type("EServiceDefHTTP", model)
    dialogues = get_children_of_type("Dialogue", model)
    access_control = get_children_of_type("AccessControlDef", model)
    connectors = get_children_of_type("Slack", model) + get_children_of_type("Telegram", model)
    print(f"Trainable Entities: {[e.name for e in entities]}")
    print(f"Synonyms: {[s.name for s in synonyms]}")
    print(f"Global Slots: {[gs.name for gs in gslots]}")
    print(f"Intents: {[i.name for i in intents]}")
    print(f"Events: {[e.name for e in events]}")
    print(f"External Services: {[e.name for e in eservices]}")
    print(f"Dialogues: {[d.name for d in dialogues]}")
    print(f"Access Control: {True if access_control else False}")
    print(f"Connectors: {[c.name for c in connectors]}")


@language("dflow", "*.dflow")
def dflow_language():
    "DFlow DSL for building intent-based Virtual Assistants (VAs)"
    mm = get_metamodel()
    return mm

class Event(BaseModel):
    type = 'Event'
    name: str
    uri: str

class Response(BaseModel):
    type: str
    name: str

class Param(BaseModel):
    type: str
    name: str
    source: str
    
class Form(Response):
    type = 'Form'
    params: list[Param]

class Action(BaseModel):
    type: str
    keyword: str
    content: str
    response_filters: Optional[str]
    roles: Optional[str]
    
class ActionGroup(Response):
    type = 'ActionGroup'
    actions: list[Action]
    
class Dialogue(BaseModel):
    name: str
    triggers: list[str]
    responses: list[Response]

def merge_models(raw_models: List[Any], output: bool = False):
    sections = [
        'entities',
        'synonyms',
        'gslots',
        'triggers',
        'dialogues',
        'eservices'
    ]

    # Parse models
    mm = get_metamodel()
    parsed_models = [mm.model_from_str(file) for file in  raw_models]

    merge_result = {section: [] for section in sections}

    for parsed_model, raw_model in zip(parsed_models, raw_models):
        dialoguePool: list[Dialogue] = []
        for dialogue in parsed_model.dialogues:
            dialoguePool.append(create_dialogue_template_object(dialogue, raw_model))
        
        for trigger in parsed_model.triggers:
            trigger_dialogue = find_trigger_dialogue(dialoguePool, trigger.name)
            if trigger.__class__.__name__ == 'Intent':
                phrases = extract_phrases(raw_model, trigger.name)
                
                # Intent similarity check
                similar_intent = find_similar_intent(phrases, merge_result['triggers'])
                if similar_intent:
                    # Response similarity check
                    similar_intent_dialogue = find_trigger_dialogue(merge_result['dialogues'], similar_intent.name)
                    if are_dialogues_similar(trigger_dialogue, similar_intent_dialogue):
                        similar_intent.phrases = list(set(similar_intent.phrases + phrases))
                    else:
                        raise Exception('Similar intents but different dialogues.')                                   
                else:
                    merge_result['triggers'].append(Trigger(name=trigger.name, phrases=phrases))
                    merge_result['dialogues'].append(trigger_dialogue)
                        
                    
            elif trigger.__class__.__name__ == 'Event':
                merge_result['triggers'].append(Event(name=trigger.name, uri=trigger.uri))
                merge_result['dialogues'].append(trigger_dialogue)
                
        for eservice in parsed_model.eservices:
            merge_result['eservices'].append(eservice)       
            
        for entities in parsed_model.entities:
            merge_result['entities'].append(entities)   
            
        for synonyms in parsed_model.synonyms:
            merge_result['synonyms'].append(synonyms)   
            
        for gslots in parsed_model.gslots:
            merge_result['gslots'].append(gslots)   

    TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE_DIR))
    template = jinja_env.get_template('merged.dflow.jinja')

    return template.render(merge_result)

def extract_phrases(raw_model: str, intent_name: str) -> List[str]:
    phrases = re.search(f'Intent {intent_name}([\s\S]*?)end', raw_model).group(1)
    phrasesList = phrases.split(',\n')
    return list(map(str.strip, phrasesList))

def find_similar_intent(phrases: List[str], triggers: List[Union[Trigger, Event]]) -> Optional[Trigger]:
    """
    Find and return a similar intent if it exists
    
    Args: 
        phrases: the List of strings from the new intent that I am comparing
        triggers: the List of Triggers in which to look for similar intent
        
    Returns: 
        Trigger: similar intent if it exists
        None: if there isn't a similar intent
    """
    for trigger in triggers:
        if trigger.type=='Intent' and are_lists_similar(phrases, trigger.phrases):
            return trigger
    return None

def find_trigger_dialogue(dialogues: list[Dialogue], trigger_name: str) -> Dialogue:
    """
    Find dialogue that corresponds to intent name
    """
    for dialogue in dialogues:
        if trigger_name in dialogue.triggers:
            return dialogue
    raise Exception(f'No dialogue corresponds to intent: { trigger_name }')

def create_dialogue_template_object(dialogue, raw_model: str) -> Dialogue:
    action_types = [
        'SpeakAction',
        'FireEventAction',
        'EServiceCallHTTP',
        'SetFormSlot',
        'SetGlobalSlot'
    ]
    actions_taken = {action_type: 0 for action_type in action_types}
    responses = []
    for response in dialogue.responses:
        response_type = response.__class__.__name__   #Get if it is Form or Action
        raw_response = extract_raw_response(raw_model, response.name, response_type)
        if response_type == 'Form':
            params = []
            for i, param in enumerate(response.params):
                next_param = response.params[i + 1] if i + 1 < len(response.params) else None
                params.append(Param(
                    type=param.type,
                    name=param.name,
                    source=extract_param_source(raw_response, param, next_param)
                ))
            responses.append(Form(
                name=response.name,
                params=params
            ))
        elif response_type == 'ActionGroup':
            actions = []
            for action in response.actions:
                action_type = action.__class__.__name__
                keyword, content, response_filters, roles = extract_action_details(raw_response, action_type, actions_taken[action_type])
                actions_taken[action_type] += 1
                actions.append(Action(
                    type=action_type,
                    keyword=keyword,
                    content=content,
                    response_filters=response_filters,
                    roles=roles
                ))
            responses.append(ActionGroup(
                name=response.name,
                actions=actions
            ))
            
        
    return Dialogue(
        name=dialogue.name,
        triggers=list(map(lambda trigger: trigger.name, dialogue.onTrigger)),
        responses=responses
    )

def extract_action_details(raw_ActionGroup: str, action_type: str, action_type_index: int) -> tuple[str, str, Optional[str]]:
    '''
    Extracts action_keyword, content and roles from Action
    '''
    action_keyword = None
    if action_type == 'SpeakAction':
        action_keyword = 'Speak'
    elif action_type == 'FireEventAction':
        action_keyword = 'FireEvent'
    elif action_type == 'SetFormSlot':
        action_keyword = 'SetFSlot'
    elif action_type == 'SetGlobalSlot':
        action_keyword = 'SetGSlot'
    elif action_type == 'EServiceCallHTTP':
        action_keyword = r'\b(?!(?:Speak|FireEvent|SetFSlot|SetGSlot)\b)\w+'
 
    action_match = list(re.finditer(f'({action_keyword})\(([\s\S]+?)\)(?:\[([\s\S]+?)\])?(?:\[([\s\S]+?)\])?', raw_ActionGroup))[action_type_index]
    action_keyword = action_match.group(1)
    content = action_match.group(2)
    response_filters, roles = None, None
    if len(action_match.groups()) == 3:
        roles = action_match.group(3) #could be either roles or response_filters
    elif len(action_match.groups()) == 4:
        response_filters = action_match.group(3) 
        roles = action_match.group(4) 
    return action_keyword, content, response_filters, roles

def extract_raw_response(raw_model: str, response_name: str, response_type: str) -> str:
    return re.search(f'{response_type} {response_name}[\s\S]+?end', raw_model).group(0)

def extract_param_source(raw_response: str, param, next_param: Optional[Any]) -> str:
    return re.search(f'{param.name}: {param.type} = ([\s\S]+?)\s*{next_param.name if next_param else "end"}', raw_response).group(1)
    
def are_dialogues_similar(dialogue1: Dialogue, dialogue2: Dialogue) -> bool:
    if len(dialogue1.responses) != len(dialogue2.responses):
        print('Number of responses does not match')
        return False
    for response1_i, response2_i in zip(dialogue1.responses, dialogue2.responses):
        if response1_i.type != response2_i.type:
            print('Mismatch in response types: expected corresponding types (Form/ActionGroup).')
            return False
    for response1_i, response2_i in zip(dialogue1.responses, dialogue2.responses):
        if response1_i.type == 'Form' and len(response1_i.params) != len(response2_i.params): # obviously response2_i is also Form
            print('Number of params in Forms does not match')
            return False
        if response1_i.type == 'ActionGroup' and len(response1_i.actions) != len(response2_i.actions): # obviously response2_i is also ActionGroup
            print('Number of actions in ActionGroups does not match')
            return False
    for response1_i, response2_i in zip(dialogue1.responses, dialogue2.responses):
        if response1_i.type == 'Form':
            for param1_i, param2_i in zip(response1_i.params, response2_i.params):
                if param1_i.type != param2_i.type:
                    print('Mismatch in param types in Forms: expected corresponding types (int, float, str, bool, list, dict).')
                    return False            
        if response1_i.type == 'ActionGroup':
            for action1_i, action2_i in zip(response1_i.actions, response2_i.actions):
                if action1_i.type != action2_i.type:
                    print('Mismatch in action types in ActionGroups: expected corresponding types (SpeakAction, FireEventAction, SetFormSlot, SetGlobalSlot, EServiceCallHTTP)')
                    return False      
    for response1_i, response2_i in zip(dialogue1.responses, dialogue2.responses):
        if response1_i.type == 'Form':
            for param1_i, param2_i in zip(response1_i.params, response2_i.params):
                hri1 = extract_hri(param1_i)
                hri2 = extract_hri(param2_i)
                #check the similarity between param sources
                if hri1 and hri2 and not are_lists_similar([hri1], [hri2]):
                    print('The HRIs inside the forms are not similar')
                    return False              
        if response1_i.type == 'ActionGroup':
            for action1_i, action2_i in zip(response1_i.actions, response2_i.actions):
                if action1_i.type == 'SpeakAction' and not are_lists_similar(action1_i.content, action2_i.content):  #they have the same type
                    print('The Speaks inside the ActionGroups are not similar')
                    return False     
    return True

def extract_hri(param: Param)-> Optional[str]:
    match = re.match(r'HRI\(([\S\s]+)\)', param.source)
    return match.group(1) if match else None

