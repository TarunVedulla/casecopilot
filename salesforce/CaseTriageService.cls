/**
 * CaseTriageService
 * Sends a new email-to-case record to the LLM with the active system prompt
 * (stored in Custom Metadata AI_Prompt__mdt) and writes the structured answer
 * back to AI_* fields on the Case. Nothing is sent to the customer: the agent
 * reviews the suggestion in the console and decides.
 *
 * Invoked from the record-triggered flow "Case_AI_Triage" (after insert,
 * Origin = 'Email') through the invocable method below.
 */
public with sharing class CaseTriageService implements Queueable, Database.AllowsCallouts {

    private static final String ENDPOINT = 'callout:LLM_API/openai/v1/chat/completions';
    private final List<Id> caseIds;

    public CaseTriageService(List<Id> caseIds) {
        this.caseIds = caseIds;
    }

    @InvocableMethod(label='Run AI Case Triage' description='Classifies new email cases with the LLM')
    public static void triage(List<Id> caseIds) {
        System.enqueueJob(new CaseTriageService(caseIds));
    }

    public void execute(QueueableContext ctx) {
        AI_Prompt__mdt prompt = [
            SELECT Version__c, Model__c, System_Prompt__c
            FROM AI_Prompt__mdt
            WHERE Active__c = true AND Use_Case__c = 'CASE_TRIAGE'
            LIMIT 1
        ];
        List<Case> updates = new List<Case>();
        // Queueable allows 100 callouts; the flow passes small batches
        for (Case c : [SELECT Id, Subject, Description, Contact.Sender_Type__c
                       FROM Case WHERE Id IN :caseIds LIMIT 50]) {
            updates.add(classify(c, prompt));
        }
        update updates;
    }

    @TestVisible
    private static Case classify(Case c, AI_Prompt__mdt prompt) {
        Case result = new Case(Id = c.Id, AI_Prompt_Version__c = prompt.Version__c);
        String userMsg = 'Sender type (from Salesforce contact record): '
            + (c.Contact?.Sender_Type__c == null ? 'unknown' : c.Contact.Sender_Type__c)
            + '\nSubject: ' + c.Subject + '\n\n' + c.Description;

        Map<String, Object> body = new Map<String, Object>{
            'model' => prompt.Model__c,
            'temperature' => 0,
            'max_tokens' => 900,
            'messages' => new List<Object>{
                new Map<String, String>{ 'role' => 'system', 'content' => prompt.System_Prompt__c },
                new Map<String, String>{ 'role' => 'user',   'content' => userMsg }
            }
        };
        HttpRequest req = new HttpRequest();
        req.setEndpoint(ENDPOINT);
        req.setMethod('POST');
        req.setHeader('Content-Type', 'application/json');
        req.setTimeout(60000);
        req.setBody(JSON.serialize(body));

        try {
            HttpResponse res = new Http().send(req);
            if (res.getStatusCode() != 200) {
                return fallback(result, 'HTTP ' + res.getStatusCode());
            }
            Map<String, Object> outer = (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
            Map<String, Object> msg = (Map<String, Object>)
                ((Map<String, Object>) ((List<Object>) outer.get('choices'))[0]).get('message');
            String content = ((String) msg.get('content')).replaceAll('^```(json)?|```$', '').trim();
            Map<String, Object> ai = (Map<String, Object>) JSON.deserializeUntyped(content);

            result.AI_Category__c        = (String) ai.get('category');
            result.AI_Priority__c        = (String) ai.get('priority');
            result.AI_Adverse_Event__c   = (Boolean) ai.get('adverse_event');
            result.AI_Review_Required__c = (Boolean) ai.get('requires_human_review');
            result.AI_Review_Reason__c   = (String) ai.get('review_reason');
            result.AI_Summary__c         = (String) ai.get('summary');
            result.AI_Template__c        = (String) ai.get('suggested_template');
            result.AI_Draft_Reply__c     = (String) ai.get('draft_reply');
            result.AI_Status__c          = 'Suggested';
            // Safety net independent of the model: any AE goes to Drug Safety
            if (result.AI_Adverse_Event__c == true) {
                result.Priority = 'High';
                List<Group> q = [SELECT Id FROM Group WHERE Type = 'Queue'
                                 AND DeveloperName = 'Drug_Safety' LIMIT 1];
                if (!q.isEmpty()) {
                    result.OwnerId = q[0].Id;
                }
            }
        } catch (Exception e) {
            return fallback(result, e.getMessage());
        }
        return result;
    }

    /** If the model fails, the case stays in the normal manual queue. */
    private static Case fallback(Case result, String reason) {
        result.AI_Status__c = 'Failed';
        result.AI_Review_Required__c = true;
        result.AI_Review_Reason__c = 'AI triage unavailable: ' + reason.left(200);
        return result;
    }
}
