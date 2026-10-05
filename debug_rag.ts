import * as ragService from './backend/services/rag.service.js';

const symptoms9 = ['joint pain', 'joint swelling', 'wrist pain', 'hand or finger pain', 'ache all over', 'peripheral edema'];
const all9 = ragService.searchDiseases(symptoms9);
const raIdx = all9.findIndex(r => r.disease === 'Rheumatoid Arthritis');
console.log('RA rank:', raIdx === -1 ? 'NOT IN TOP 10' : raIdx + 1, all9[raIdx] ? '(score: ' + all9[raIdx].relevance.toFixed(1) + '%)' : '');
console.log('Top 10 for TC09:');
all9.slice(0, 10).forEach((r, i) => console.log('  ' + (i+1) + '. ' + r.disease + ' (' + r.relevance.toFixed(1) + '%)'));

const symptoms15 = ['urinary frequency', 'penile burning sensation', 'dizziness'];
const all15 = ragService.searchDiseases(symptoms15);
console.log('\nTop 10 for TC15 (your case):');
all15.slice(0, 10).forEach((r, i) => console.log('  ' + (i+1) + '. ' + r.disease + ' (' + r.relevance.toFixed(1) + '%)'));

// Check what "urinary frequency" and "penile burning sensation" match in KB
const val = ragService.validateSymptomMatches(['urinary frequency', 'penile burning sensation', 'dizziness'], 'Cystitis');
console.log('\nCystitis validation for TC15:', JSON.stringify(val, null, 2));
