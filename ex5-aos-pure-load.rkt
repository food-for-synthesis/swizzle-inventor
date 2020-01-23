#|
 | Copyright (c) 2018-2019, University of California, Berkeley.
 | Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 |
 | Redistribution and use in source and binary forms, with or without 
 | modification, are permitted provided that the following conditions are met:
 |
 | 1. Redistributions of source code must retain the above copyright notice, 
 | this list of conditions and the following disclaimer.
 |
 | 2. Redistributions in binary form must reproduce the above copyright notice, 
 | this list of conditions and the following disclaimer in the documentation 
 | and/or other materials provided with the distribution.
 |
 | THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 | AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 | IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 | ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 | LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 | CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 | SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 | INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 | CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 | ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 | POSSIBILITY OF SUCH DAMAGE.
 |#

#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(output-smt #t) 

(define struct-size 5)
(define n-block 1)

(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size (* 1 warpSize))
  (define array-size (* n-block block-size))
  (define I-sizes (x-y-z (* array-size struct-size)))
  (define I (create-matrix I-sizes gen-uid))
  (define O (create-matrix I-sizes))
  (define O* (create-matrix I-sizes))
  (values block-size I-sizes I O O*))

(define (run-with-warp-size spec kernel w)
  (define-values (block-size I-sizes I O O*)
  (create-IO w))

  (define c (gcd struct-size warpSize))
  (define a (/ struct-size c))
  (define b (/ warpSize c))

  (run-kernel spec (x-y-z block-size) (x-y-z n-block) I O a b c)
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) I O* a b c)
  (define ret (equal? O O*))
  ;(pretty-display `(O ,O))
  ;(pretty-display `(O* ,O*))
  ret)

(define (AOS-load-spec threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-local I I-cached
                 (x-y-z struct-size)
                 offset (x-y-z (* warpSize struct-size)) #f)
  (local-to-global I-cached O
                      (x-y-z 1) offset (x-y-z (* warpSize struct-size)) #f #:round struct-size)
  )

(define (print-vec x)
  (format "#(~a)" (string-join (for/list ([xi x]) (format "~a" xi)))))

;; Sketch that uses column-row-column shuffles. 
(define (AOS-load-sketch threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define O-cached (create-matrix-local (x-y-z struct-size)))
  
  (define localId (modulo (get-x threadId) 32))
  (define offset (* struct-size (- (+ (* blockID blockDim) (get-x threadId)) localId)))
  
  (global-to-local I I-cached
                 (x-y-z 1)
                 offset
                 (x-y-z (* warpSize struct-size)) #f #:round struct-size)

  ;; column shuffle
  (define I-cached2 (permute-vector I-cached struct-size
                                    (lambda (i) (?sw-xform i struct-size localId warpSize))))

  (define learning #t) ; if true, we are solving for where the values from `als` are in the solution
  (define pruning  #f) ; if true, we are using the solution to prune

  ; these are manually selected "random" input values, smalled number often works better 

  (define vals     (list  13 27 )) ;  93 155 17  111 53  99  141 77  33  16  64))
  (define rows1-?? (list (??)(??)(??)(??)(??)(??)(??)(??)(??)(??)(??)(??)(??)))
  (define cols1-?? (list (??)(??)(??)(??)(??)(??)(??)(??)(??)(??)(??)(??)(??)))
  (define rows2-?? (list (??)(??)(??)(??)(??)(??)(??)(??)(??)(??)(??)(??)(??)))
  (define cols2-?? (list (??)(??)(??)(??)(??)(??)(??)(??)(??)(??)(??)(??)(??)))

   (define rows1-sol
     (list 2 1 (??) (??) (??) (??) (??) (??) (??) (??) (??) (??) (??)))
   (define cols1-sol
     (list 12 26 (??) (??) (??) (??) (??) (??) (??) (??) (??) (??) (??)))
   (define rows2-sol
     (list 2 1 (??) (??) (??) (??) (??) (??) (??) (??) (??) (??) (??)))
   (define cols2-sol
     (list 2 5 (??) (??) (??) (??) (??) (??) (??) (??) (??) (??) (??)))

  (let ([V I-cached2])
    (when pruning
      (for ([v vals]
            [r rows1-sol]
            [c cols1-sol])
        (assert (equal? v  (vector-ref (vector-ref V c) r)))))

    (when learning      
      (for ([v vals]
            [r rows1-??]
            [c cols1-??])
        (assert (equal? v  (vector-ref (vector-ref V c) r))))))

  ;; 
  ;; row shuffle
  ;;
  (for ([i struct-size])
    (let* ([lane (?sw-xform localId warpSize i struct-size)]
           [x (shfl (get I-cached2 (@dup i)) lane)]
           )
      (set O-cached (@dup i) x))
    )
  
  (let ([V O-cached])
    (when pruning
      (for ([v vals]
            [r rows2-sol]
            [c cols2-sol])
        (assert (equal? v  (vector-ref (vector-ref V c) r)))))

    (when learning      
      (for ([v vals]
            [r rows2-??]
            [c cols2-??])
        (assert (equal? v  (vector-ref (vector-ref V c) r))))))
  
  ;; column shuffle
  (define O-cached2 (permute-vector O-cached struct-size
                                    (lambda (i) (?sw-xform i struct-size localId warpSize))))
  

  (local-to-global O-cached2 O
                      (x-y-z 1)
                      offset
                      (x-y-z (* warpSize struct-size)) #f #:round struct-size)
  )


;; Sketch that uses column-row-column shuffles.  
;; This variant synthesizes one pruning rule.
;; The rule is given by the solution to c1, r1, c2, r2. 

(define-symbolic v1 v2 c1 r1 c2 r2 integer?)
(define pruning-holes (list v1 v2 c1 r1 c2 r2))
(define (legal-value? v) 
  (and (< 0 v) (<= v (* struct-size warpSize))))
(assert (legal-value? v1))
(assert (equal? 13 v1))
(assert (legal-value? v2))
; (assert (equal? 27 v2)) ; TODO: enabling this leads to no solutions => investigate
; (assert (distinct? v1 v2))
(assert (or (distinct? c1 c2)(distinct? r1 r2)))
(define synth-cond #t) ; for storing the correctness condition 

;; Note: This particular pruning is weaker than what we use.
;; In particular, this finds partial assignments that are not part of any correct solution. 
;; That is, it finds a pair of intermediate elements e1=c1,r1 and e2=c2,r2 such that 
;;  e1,e2 cannot be reached from the source 
;;    OR 
;;  e1,e2 cannot reach the target
;; In contrast, we use only the second condition.
(define (AOS-load-sketch-prune-failed threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define O-cached (create-matrix-local (x-y-z struct-size)))
  
  (define localId (modulo (get-x threadId) 32))
  (define offset (* struct-size (- (+ (* blockID blockDim) (get-x threadId)) localId)))
  
  (global-to-local I I-cached
                 (x-y-z 1)
                 offset
                 (x-y-z (* warpSize struct-size)) #f #:round struct-size)

  ;; column shuffle
  (define I-cached2 (permute-vector I-cached struct-size
                                    (lambda (i) (?sw-xform i struct-size localId warpSize))))

  
  ;; 
  ;; row shuffle
  ;;
  (for ([i struct-size])
    (let* ([lane (?sw-xform localId warpSize i struct-size)]
           [x (shfl (get I-cached2 (@dup i)) lane)]
           )
      (set O-cached (@dup i) x))
    )
  
  (set! synth-cond 
    (let ([V O-cached])
      (! (&& 
                 (equal? 13  (vector-ref (vector-ref V c1) r1))
                 ; (equal? 27  (vector-ref (vector-ref V c2) r2))
                 ))))


  ;; column shuffle
  (define O-cached2 (permute-vector O-cached struct-size
                                    (lambda (i) (?sw-xform i struct-size localId warpSize))))
  

  (local-to-global O-cached2 O
                      (x-y-z 1)
                      offset
                      (x-y-z (* warpSize struct-size)) #f #:round struct-size)
  )



(define (havoc-input-value)
  (define-symbolic* input integer?)
  (assert (legal-value? input))
  input
  )
;; Note: This particular pruning is equivalent to what we use. 
(define (AOS-load-sketch-prune-failed-2 threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size) havoc-input-value))
  (define O-cached (create-matrix-local (x-y-z struct-size) havoc-input-value))
  
  (define localId (modulo (get-x threadId) 32))
  (define offset (* struct-size (- (+ (* blockID blockDim) (get-x threadId)) localId)))

  (global-to-local I I-cached
                 (x-y-z 1)
                 offset
                 (x-y-z (* warpSize struct-size)) #f #:round struct-size)
  
  ;; column shuffle
  (define I-cached2 (permute-vector I-cached struct-size
                                    (lambda (i) (?sw-xform i struct-size localId warpSize))))

  ;; By havocing the intermediate value, the synthesized pruning rule will assume
  ;; that the traced values (13 and 27) can be supplied in any elements. The synthesis 
  ;; determines if there is a pair of elements from which 13 and 27 can't reach the target.
  
  #;(set! I-cached2 (create-matrix-local (x-y-z struct-size) havoc-input-value))
  #;(set! synth-cond 
    (let ([V I-cached2])
      (! (&& 
                 (equal? 13  (vector-ref (vector-ref V c1) r1))
                 (equal? 27  (vector-ref (vector-ref V c2) r2))
                 ))))

  ;; 
  ;; row shuffle
  ;;
  (for ([i struct-size])
    (let* ([lane (?sw-xform localId warpSize i struct-size)]
           [x (shfl (get I-cached2 (@dup i)) lane)]
           )
      (set O-cached (@dup i) x))
    )

  #;(set! O-cached (create-matrix-local (x-y-z struct-size) havoc-input-value))
  #;(set! synth-cond 
    (let ([V O-cached])
      (! (&& 
                 (equal? 13  (vector-ref (vector-ref V c1) r1))
                 (equal? 27  (vector-ref (vector-ref V c2) r2))
                 ))))

  ;; column shuffle
  (define O-cached2 (permute-vector O-cached struct-size
                                    (lambda (i) (?sw-xform i struct-size localId warpSize))))
  

  (local-to-global O-cached2 O
                      (x-y-z 1)
                      offset
                      (x-y-z (* warpSize struct-size)) #f #:round struct-size)
  )

;; Sketch that uses row-column-row shuffles.
(define (AOS-load-rcr-sketch threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  
  (define localId (modulo (get-x threadId) 32))
  (define offset (* struct-size (- (+ (* blockID blockDim) (get-x threadId)) localId)))

  ;; load with (row) shuffle
  (global-to-local
   I
   I-cached
   (x-y-z 1)
   offset
   (x-y-z (* warpSize struct-size))
   #f #:round struct-size
   #:shfl (lambda (localId i) (?sw-xform localId warpSize i struct-size)))

  ;; column shuffle
  (define O-cached (permute-vector I-cached struct-size
                                   (lambda (i) (?sw-xform i struct-size localId warpSize))))

  ;; store with (row) shuffle
  (local-to-global
   O-cached
   O
   (x-y-z 1)
   offset
   (x-y-z (* warpSize struct-size))
   #f #:round struct-size
   #:shfl (lambda (localId i)
            (?sw-xform localId warpSize i struct-size)))
  )

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-load-spec AOS-load-sketch w)])
      (pretty-display `(test ,w ,ret))))
  )
;(test)

(define (synthesis)
  (pretty-display "solving...")
  (assert (andmap (lambda (w) (run-with-warp-size AOS-load-spec AOS-load-sketch w))
                                           (list 32)))
  (define cost (get-cost))
  
  ; (define sol (time (optimize #:minimize (list cost) #:guarantee (assert #t))))

  (define sol (time (solve (assert #t))))
  (print sol)
  (define this-cost (evaluate cost sol))
  (print-forms sol)
  (pretty-display `(cost ,this-cost))
  )

(define (synthesize-pruning)
  (pretty-display "solving pruning...")

  ; The solutions satisfying these assumptions produce the correct values 
  ; at the target, assuming that the input is an arbitrary value 
  ; (see how I-cached is havoced).
  (define assumptions (with-asserts-only 
      (assert (andmap (lambda (w) (run-with-warp-size AOS-load-spec 
                                                  AOS-load-sketch-prune-failed-2
                                                  w))
                                           (list 32)))))

  ; the forall symbols for synthesis are 
  ;   -- holes for the swizzles (we iterate over all possible candidate swizzles), and
  ;   -- values in the input matrix (this assumes that any value can be supplied) 
  ; the holes for the pruning rule are kept apart, and are solved in (synthesize)

  (define forall-symbols (remv* pruning-holes (symbolics assumptions)))

  ; The synth-condition is the pruning rule. It is set in `AOS-load-sketch-prune-failed-2`
  (define sol (time (synthesize #:forall forall-symbols
                                #:assume (assert (apply && assumptions))
                                #:guarantee (assert synth-cond)
                                )))
  (println sol)
  ; (print-forms sol)
  )

(define (verify-pruning)
  (pretty-display "verifying pruning...")
  (define-values (cond as) (with-asserts 
     (andmap (lambda (w) (run-with-warp-size AOS-load-spec 
                                                  AOS-load-sketch-prune-failed-2 
                                                  w))
                                           (list 32))))
  (define inputs (remv* pruning-holes (symbolics as)))
  (define sol (time (synthesize #:forall inputs
                                #:assume (assert (apply && as))
                                #:guarantee (assert cond)
                                )))
  (println sol)
  ; (print-forms sol)
  )

(define t0 (current-seconds))
(synthesize-pruning)
(define t1 (current-seconds))
(- t1 t0)
