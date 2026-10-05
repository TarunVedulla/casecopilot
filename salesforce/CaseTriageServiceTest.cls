@IsTest
private class CaseTriageServiceTest {

    private class LlmMock implements HttpCalloutMock {
        private final Integer status;
        private final String content;
        LlmMock(Integer status, String content) { this.status = status; this.content = content; }
        public HttpResponse respond(HttpRequest req) {
            HttpResponse res = new HttpResponse();
            res.setStatusCode(status);
            res.setBody(JSON.serialize(new Map<String, Object>{
                'choices' => new List<Object>{ new Map<String, Object>{
                    'message' => new Map<String, Object>{ 'role' => 'assistant', 'content' => content } } }
            }));
            return res;
        }
    }

    private static AI_Prompt__mdt prompt() {
        return new AI_Prompt__mdt(Version__c = 'v3', Model__c = 'test-model', System_Prompt__c = 'test');
    }

    @IsTest
    static void adverseEventIsFlaggedAndRouted() {
        Case c = new Case(Subject = 'Headache', Description = 'Headache after injection', Origin = 'Email');
        insert c;
        String payload = '{"category":"ADVERSE_EVENT","priority":"HIGH","adverse_event":true,'
            + '"requires_human_review":true,"review_reason":"AE","summary":"s",'
            + '"suggested_template":"TPL_AE_ACK","draft_reply":"Dear Sir or Madam"}';
        Test.setMock(HttpCalloutMock.class, new LlmMock(200, payload));
        Test.startTest();
        Case r = CaseTriageService.classify(c, prompt());
        Test.stopTest();
        System.assertEquals('ADVERSE_EVENT', r.AI_Category__c);
        System.assertEquals(true, r.AI_Adverse_Event__c);
        System.assertEquals('High', r.Priority);
        System.assertEquals('Suggested', r.AI_Status__c);
    }

    @IsTest
    static void invalidJsonFallsBackToManual() {
        Case c = new Case(Subject = 'x', Description = 'y', Origin = 'Email');
        insert c;
        Test.setMock(HttpCalloutMock.class, new LlmMock(200, 'Category: Other'));
        Test.startTest();
        Case r = CaseTriageService.classify(c, prompt());
        Test.stopTest();
        System.assertEquals('Failed', r.AI_Status__c);
        System.assertEquals(true, r.AI_Review_Required__c);
    }

    @IsTest
    static void httpErrorFallsBackToManual() {
        Case c = new Case(Subject = 'x', Description = 'y', Origin = 'Email');
        insert c;
        Test.setMock(HttpCalloutMock.class, new LlmMock(500, ''));
        Test.startTest();
        Case r = CaseTriageService.classify(c, prompt());
        Test.stopTest();
        System.assertEquals('Failed', r.AI_Status__c);
    }
}
