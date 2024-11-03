import javalang


def parse_node(node, indent=0):
    """
    Parse a node and return the signature.
    """
    if isinstance(node, javalang.tree.FieldDeclaration):
        declaration = " " * indent
        if len(node.annotations) > 0:
            for i in node.annotations:
                declaration += parse_annotation(i) + "\n"
        if node.modifiers:
            declaration += " ".join(node.modifiers) + " "
        declaration += parse_reference_type(node.type) + " "
        declarator_names = [i.name for i in node.declarators]
        declaration += ", ".join(declarator_names)
        return declaration + ";"

    if isinstance(node, javalang.tree.MethodDeclaration):
        declaration = " " * indent
        if len(node.annotations) > 0:
            for i in node.annotations:
                declaration += parse_annotation(i) + "\n"
        if node.modifiers:
            declaration += " ".join(node.modifiers) + " "
        declaration += parse_reference_type(node.return_type) + " "
        declaration += node.name + "("
        declaration += ", ".join([parse_formal_parameter(i) for i in node.parameters]) + ")"
        if node.throws:
            declaration += "throws " + ", ".join(node.throws)
        return declaration + ";"

    if isinstance(node, javalang.tree.ConstructorDeclaration):
        declaration = " " * indent
        if len(node.annotations) > 0:
            for i in node.annotations:
                declaration += parse_annotation(i) + "\n"
        if node.modifiers:
            declaration += " ".join(node.modifiers) + " "
        declaration += node.name + "("
        declaration += ", ".join([parse_formal_parameter(i) for i in node.parameters]) + ")"
        return declaration + ";"

    if isinstance(node, javalang.tree.EnumDeclaration):
        declaration = " " * indent
        if node.modifiers:
            declaration += " ".join(node.modifiers) + " "
        declaration += "enum " + node.name + "{"
        declaration += ", ".join(i.name for i in node.body.constants) + "}"
        return declaration + ""

    if isinstance(node, javalang.tree.ClassDeclaration):
        declaration = " " * indent
        if node.modifiers:
            declaration += " ".join(node.modifiers) + " "
        declaration += "class " + node.name
        if node.extends:
            declaration += " extends " + parse_reference_type(node.extends)
        if node.implements:
            declaration += " implements " + ", ".join([parse_reference_type(i) for i in node.implements])
        declaration += " {\n"
        if node.body:
            for i in node.body:
                declaration += parse_node(i, indent=indent + 4)
        declaration += " " * indent + "}"
        return declaration
    return ""

def parse_method_declaration(node):
    if isinstance(node, javalang.tree.MethodDeclaration):
        params = [parse_formal_parameter(i) for i in node.parameters]

        def f(node):
            if hasattr(node, 'children'):
                for child in node.children:
                    yield from f(child)
            if isinstance(node, list):
                for child in node:
                    yield from f(child)
            if isinstance(node, javalang.tree.MethodInvocation):
                yield parse_method_invocation(node)

        invocation = set(f(node))
        return params, invocation

# def parse_method_invocation(node):
#     if isinstance(node, javalang.tree.MethodInvocation):
#         ret = node.member + "("
#         for param in node.arguments:
#             ret += parse_reference(param) + ", "
#         ret = ret[:-2] + ")"
#         return ret

def parse_method_invocation(node):
    if isinstance(node, javalang.tree.MethodInvocation):
        return node.member

def parse_reference(refer):
    if isinstance(refer, javalang.tree.MemberReference):
        return refer.member
    if isinstance(refer, javalang.tree.Literal):
        return refer.value
def parse_element_value_pair(node):
    if isinstance(node, javalang.tree.ElementValuePair):
        try:
            ret = node.name + " = " + node.value.value
        except:
            ret = node.name
        return ret

def parse_annotation(node):
    if isinstance(node, javalang.tree.Annotation):
        ret = "@"
        ret += node.name
        if isinstance(node.element, javalang.tree.Literal):
            ret += node.element.value
        # elif isinstance(node.element, javalang.tree.ElementArrayValue):
        #     ret += str(node.element.values)
        elif isinstance(node.element, list) and len(node.element) > 0:
            ret += "("
            for i in node.element:
                ret += parse_element_value_pair(i)
                ret += ", "
            ret = ret[:-2]
            ret += ")"
        else:
            ret += ""
        return ret

def parse_reference_type(node):
    """
    A fucking helpful method to parse the fucking reference type.
    """
    declaration = ""
    if node == None:
        return "void"
    if isinstance(node, javalang.tree.ReferenceType):
        args = node.arguments
        if args is None:
            # arg_name = args.name
            return node.name
        else:
            arg_name = [parse_reference_type(i.type) for i in args]
        declaration += node.name + ("<" + ", ".join(arg_name) + ">" if args else "")
    else:
        declaration += parse_basic_type(node)
    return declaration


def parse_formal_parameter(node):
    """
    A fucking helpful method to parse the fucking formal parameter type.
    """
    if isinstance(node, javalang.tree.FormalParameter):
        ret = ""
        if len(node.annotations) > 0:
            for i in node.annotations:
                ret += parse_annotation(i)
        return parse_reference_type(node.type) + " " + node.name


def parse_basic_type(node):
    """
    A fucking helpful method to parse the fucking basic type.
    """
    declaration = ""
    if isinstance(node, javalang.tree.BasicType):
        declaration += node.name
        for i in node.dimensions:
            declaration += "[]"
    return declaration


if __name__ == "__main__":

    with open("Queue.java", 'r') as f:
        file = f.read()

    # print(file)
    tree = javalang.parse.parse(file)
    # print(tree)

    # Import info
    import_info = [x.path for x in tree.imports]
    print("import info:")
    print(import_info)

    class_declaration = tree.types[0]
    # class name
    print("class name:")
    print(class_declaration.name)

    class_body = class_declaration.body  # -> list
    # Construct info
    print("construct info: ")
    construct_list = []
    for item in class_body:
        if type(item) == javalang.tree.ConstructorDeclaration:
            construct_list.append(parse_node(item))

    for i in construct_list:
        print(i)

    # properties info
    print("properties info: ")
    properties_list = []
    for item in class_body:
        if type(item) == javalang.tree.FieldDeclaration:
            properties_list.append(parse_node(item))

    for i in properties_list:
        print(i)

    # methods info(contain method call)
    print("methods info: ")
    methods_info = []
    methods_params = []
    methods_invoke = []
    for item in class_body:
        if type(item) == javalang.tree.MethodDeclaration:
            # print("yes")
            methods_info.append(parse_node(item))
            params, invokes = parse_method_declaration(item)
            methods_params.append(params)
            methods_invoke.append(invokes)

    for i in range(len(methods_info)):
        print(methods_info[i])
        print(methods_params[i])
        print(methods_invoke[i])
        print()


__all__ = ["parse_node", "parse_method_declaration", "parse_method_invocation"]
