@{
    # Excluir reglas específicas
    ExcludeRules = @(
        'PSUseApprovedVerbs',  # Permitir verbos no aprobados en scripts locales
        'PSAvoidUsingCmdletAliases'  # Permitir alias como cd, ls, etc.
    )
    
    # Severidad
    Severity = @('Error', 'Warning')
    
    # Incluir reglas por defecto excepto las excluidas
    IncludeDefaultRules = $true
}
